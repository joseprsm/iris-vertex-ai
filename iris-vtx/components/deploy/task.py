import click

from google.cloud import aiplatform as aip

ENDPOINT_NAME = 'iris-deploy'
DISPLAY_NAME = 'iris-vtx'
MODEL_NAME = 'iris-vtx'


@click.command()
@click.option('--project', type=str)
@click.option('--region', type=str)
@click.option('--model-uri', type=str)
@click.option('--serving-image-uri', type=str)
def deploy(project, region, model_uri, serving_image_uri):

    endpoint = create_endpoint(project, region)

    model_upload = aip.Model.upload(
        display_name=DISPLAY_NAME,
        serving_container_image_uri=serving_image_uri,
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        serving_container_environment_variables={
            "MODEL_URI": model_uri,
        },
    )

    _ = model_upload.deploy(
        machine_type="n1-standard-4",
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )


def create_endpoint(project, region):
    endpoints = aip.Endpoint.list(
        filter='display_name="{}"'.format(ENDPOINT_NAME),
        order_by='create_time desc',
        project=project,
        location=region,
    )

    endpoint = endpoints[0] if len(endpoints) > 0 else aip.Endpoint.create(
        display_name=ENDPOINT_NAME, project=project, location=region
    )

    return endpoint


if __name__ == '__main__':
    deploy()
