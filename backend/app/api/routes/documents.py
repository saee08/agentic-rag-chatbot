from __future__ import annotations

from fastapi import APIRouter, File, Response, UploadFile, status

from app.api.deps import ContainerDep, CurrentUser, SessionDep, enforce
from app.core.exceptions import ValidationFailedError
from app.schemas import DocumentOut, ListResponse, UploadResponse, UploadResult

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_documents(user: CurrentUser, container: ContainerDep, session: SessionDep,
                     files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload one or more files. Processing happens in the background; poll the document
    status. Re-uploading identical content returns the existing document (duplicate=true)."""
    enforce(container.rate_limits.upload, f"user:{user.id}")
    if len(files) > container.settings.max_files_per_upload:
        raise ValidationFailedError(f"At most {container.settings.max_files_per_upload} files per upload.")
    results = []
    for upload in files:
        doc, duplicate, job = container.documents.store_upload(session, user.id, upload.filename, upload.file)
        if job is not None:
            container.worker.submit(job.id)
        results.append(UploadResult(document=DocumentOut.model_validate(doc), duplicate=duplicate))
    return UploadResponse(items=results)


@router.get("", response_model=ListResponse[DocumentOut])
def list_documents(user: CurrentUser, container: ContainerDep, session: SessionDep) -> ListResponse[DocumentOut]:
    return ListResponse(items=[DocumentOut.model_validate(d) for d in container.documents.list(session, user.id)])


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, user: CurrentUser, container: ContainerDep, session: SessionDep) -> DocumentOut:
    return DocumentOut.model_validate(container.documents.get(session, user.id, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, user: CurrentUser, container: ContainerDep, session: SessionDep) -> Response:
    container.documents.delete(session, user.id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def reprocess_document(document_id: str, user: CurrentUser, container: ContainerDep,
                       session: SessionDep) -> DocumentOut:
    doc, job = container.documents.reprocess(session, user.id, document_id)
    container.worker.submit(job.id)
    return DocumentOut.model_validate(doc)
