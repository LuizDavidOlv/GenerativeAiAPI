import json
from fastapi import APIRouter
from unstructured.partition.html import partition_html

router = APIRouter(
    prefix="/processor",
    tags=["Processor"]
)

@router.get("/normalize-html-file")
def normalize_html_file():
    file = 'API\DocumentFiles\html_example.html'
    elements = partition_html(filename=file)

    element_dict = [el.to_dict() for el in elements]
    example_output = json.jdumps(element_dict[11:15],ident=2)
    return example_output