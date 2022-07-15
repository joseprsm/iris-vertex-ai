import os

from kfp.v2 import compiler, dsl

from kfp.components import load_component_from_file

project_id = os.environ.get('PROJECT_ID')
pipeline_root_path = os.environ.get('PIPELINE_ROOT')


def get_component(name: str):
    return load_component_from_file(f'iris-vtx/components/{name}/component.yaml')


download_op = get_component('download')
train_op = get_component('train')
deploy_op = get_component('deploy')


@dsl.pipeline(
    name="iris-pipeline",
    pipeline_root=pipeline_root_path)
def pipeline():
    download_task = download_op()
    train_task = train_op(training_data=download_task.outputs['data'])
    deploy_task = deploy_op(model_uri=train_task.outputs['model'])


compiler.Compiler().compile(pipeline_func=pipeline, package_path='pipeline.json')
