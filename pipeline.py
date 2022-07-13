import os

from kfp.v2 import compiler, dsl

from kfp.components import load_component_from_file

from google.cloud import aiplatform as aip

project_id = os.environ.get('PROJECT_ID')
pipeline_root_path = os.environ.get('PIPELINE_ROOT')


def get_component(name: str):
    return load_component_from_file(f'components/{name}/component.yaml')


@dsl.pipeline(
    name="iris-pipeline",
    pipeline_root=pipeline_root_path)
def pipeline():
    download_op = get_component('download')()
    train = get_component('train')(training_data=download_op.outputs['data'])


aip.init(project=project_id)
compiler.Compiler().compile(pipeline_func=pipeline, package_path='pipeline.json')
