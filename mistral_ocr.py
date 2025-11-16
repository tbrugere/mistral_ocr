#!/bin/env python
"""
Written in async style for literally no reason, I don't think there's any concurrency going on. Just force of habit.
"""

import asyncio
import contextlib
from io import BufferedReader
import json
from typing import IO, AsyncIterator, BinaryIO, Iterable, Iterator, Literal, Sequence
import click
import re
from pathlib import Path
from mistralai import FileSignedURL, Mistral, UploadFileOut
import tempfile
import logging

import mistralai
from pydantic import Base64Bytes, BaseModel, ConfigDict, field_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ResponseImage(Model):
    id: str
    image_base64: Base64Bytes

    @field_validator("image_base64", mode="before")
    @classmethod
    def strip_base64_header(cls, value: str) -> str:
        base64_header = "base64,"
        header_end = value.find(base64_header) + len(base64_header)
        header = value[:header_end]
        content = value[header_end:]
        assert re.fullmatch(r"data:.*;base64,", header)
        return content

    def write_to_directory(self, directory: Path):
        img_path = directory / self.id
        img_path.write_bytes(self.image_base64)


class ResponsePage(Model):
    index: int
    markdown: str
    images: list[ResponseImage]


class ResponseBody(Model):
    pages: Sequence[ResponsePage]
    model: str

    def all_markdown(self) -> Iterator[str]:
        for page in self.pages:
            yield page.markdown

    def all_images(self) -> Iterator[ResponseImage]:
        for page in self.pages:
            for image in page.images:
                yield image


class ResponseModelResponse(Model):
    status_code: int
    body: ResponseBody

    def write(self, directory: Path, md_name: str):
        md_path = directory / f"{md_name}.md"
        with md_path.open("w") as f:
            for md in self.body.all_markdown():
                f.write(md)
                f.write("\n\n")
        for img in self.body.all_images():
            img.write_to_directory(directory)


class ResponseModel(Model):
    id: str
    custom_id: str
    response: ResponseModelResponse
    error: str | None


@contextlib.asynccontextmanager
async def upload_file_content(
    client: Mistral,
    file_name: str,
    file: BufferedReader | IO[bytes],
    purpose: Literal["ocr", "batch"] = "ocr",
) -> AsyncIterator[UploadFileOut]:
    """moves a file to mistral cloud, and deletes it on exit"""
    try:
        upload_response = await client.files.upload_async(
            file=mistralai.File(file_name=file_name, content=file.read()),
            purpose=purpose,
        )
        yield upload_response
    finally:
        try:
            await client.files.delete_async(file_id=upload_response.id)  # type: ignore
        except Exception as e:
            logging.error(
                "Couldn't delete file from mistralai cloud, you should delete it manually",
                exc_info=e,
            )
    return


@contextlib.asynccontextmanager
async def upload_file_to_url(
    client: Mistral, file_path: Path
) -> AsyncIterator[FileSignedURL]:
    with file_path.open("rb") as f:
        async with upload_file_content(
            client, file_path.name, f, purpose="ocr"
        ) as upload_response:
            signed_url = await client.files.get_signed_url_async(
                file_id=upload_response.id, expiry=3
            )
            yield signed_url


@contextlib.asynccontextmanager
async def upload_files(
    client: Mistral, file_paths: Iterable[Path]
) -> AsyncIterator[Sequence[FileSignedURL]]:
    async with contextlib.AsyncExitStack() as stack:
        with click.progressbar(file_paths, label="uploading files") as bar:
            # would be nice if we could do this in parallel since it's clearly io-bound
            # but the AsyncExitStack isn't fully safe with that
            file_uris = [
                await stack.enter_async_context(upload_file_to_url(client, path))
                for path in bar
            ]
            yield file_uris


@contextlib.contextmanager
def create_batch_file(file_uris: Iterable[FileSignedURL], file_names: Iterable[str]):
    with tempfile.TemporaryFile("w+b") as file:
        for uri, name in zip(file_uris, file_names):
            entry = {
                "custom_id": name,
                "body": {
                    "document": {
                        "type": "document_url",
                        "document_url": uri.url,
                    },
                    "include_image_base64": True,
                },
            }
            file.write(json.dumps(entry).encode())
            file.write(b"\n")
        file.seek(0)
        yield file


async def wait_for_job_to_finish(
    client: Mistral, job: mistralai.BatchJobOut, n_files: int
):
    last_known_done = 0
    retreived_job = await client.batch.jobs.get_async(job_id=job.id)
    with click.progressbar(length=n_files, label="Processing files") as pbar:
        while job.status in ["QUEUED", "RUNNING"]:
            logging.debug(job.status)
            retreived_job = await client.batch.jobs.get_async(job_id=job.id)
            known_done = retreived_job.succeeded_requests
            new_done = known_done - last_known_done
            pbar.update(new_done)
            last_known_done = new_done
    return retreived_job


@click.command()
@click.argument(
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, readable=True),
    nargs=-1,
)
@click.option(
    "--api-key",
    envvar="MISTRAL_API_KEY",
    required=True,
    help="API key for Mistral OCR service.",
    prompt=True,
    hide_input=True,
)
@click.option("--model", type=str, default="mistral-ocr-latest")
@click.option(
    "--output-file",
    "-o",
    type=click.File(
        "wb",
    ),
    required=False,
    help="save the raw result of the batch job",
)
@click.option("--resume", type=str, required=False, help="resume from job id")
def sync_main(
    file_path: tuple[Path, ...],
    api_key: str,
    model: str,
    output_file: BinaryIO | None = None,
    resume: str | None = None,
) -> None:
    asyncio.run(main(api_key, file_path, model, output_file, resume))


async def main(
    api_key: str,
    file_path: tuple[Path, ...],
    ocr_model: str,
    output_file: BinaryIO | None = None,
    resume: str | None = None,
) -> None:
    if not file_path:
        logging.error("No files provided")
        exit()
    client = Mistral(api_key)
    name_to_path = {f.name: f for f in file_path}
    if len(name_to_path) != len(file_path):
        raise ValueError(
            "For now we don't support uploading several files with the same name"
        )
    async with contextlib.AsyncExitStack() as stack:
        if not resume:
            file_uris = await stack.enter_async_context(upload_files(client, file_path))
            batch_file = stack.enter_context(
                create_batch_file(file_uris, [p.name for p in file_path])
            )
            batch_data_url = await stack.enter_async_context(
                upload_file_content(client, "batch_file", batch_file, purpose="batch")
            )
            job = client.batch.jobs.create(
                input_files=[batch_data_url.id], model=ocr_model, endpoint="/v1/ocr"
            )
            logging.info(f"running job with id {job.id}")
        else:
            job = await client.batch.jobs.get_async(job_id=resume)
        job = await wait_for_job_to_finish(client, job, len(file_path))
        assert job.output_file
        download_result = await client.files.download_async(file_id=job.output_file)
        if output_file:
            async for b in download_result.aiter_bytes():
                output_file.write(b)
        output_results_json = download_result.aiter_lines()
        output_results = [
            ResponseModel.model_validate_json(j) async for j in output_results_json
        ]

        for result in output_results:
            input_path = name_to_path[result.custom_id]
            output_path_dir = input_path.with_suffix(".ocr")
            output_path_dir.mkdir(exist_ok=True)
            result.response.write(
                output_path_dir, output_path_dir.with_suffix(".md").name
            )


if __name__ == "__main__":
    sync_main()
