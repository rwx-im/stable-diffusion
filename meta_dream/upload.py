from string import Template
import os
import io
import boto3
import base64
import html

import config

global s3
session = boto3.session.Session()
s3 = session.client(
        service_name='s3',
        aws_access_key_id=config.ACCESS_KEY_ID,
        aws_secret_access_key=config.SECRET_ACCESS_KEY,
        endpoint_url=config.ENDPOINT_URL)

def upload_image_data(image_data, job_id, prompt):
    image_data.seek(0)
    print(f"Uploading image {job_id} ..")
    print(f"{image_data}")

    html_data = open("meta_dream/image.html", "r", encoding='utf-8').read()
    template = Template(html_data)
    prompt_safe = html.escape(prompt.text)
    seed = str(prompt.seed)
    encoded = base64.b64encode(image_data.read()).decode('utf-8')
    image_data_url = 'data:image/png;base64,{}'.format(encoded)

    data = template.safe_substitute(prompt_safe=prompt_safe, seed=seed, image_data_url=image_data_url).encode('utf-8')

    filename = f"{job_id}.html"
    result = s3.upload_fileobj(io.BytesIO(data), config.BUCKET, filename, ExtraArgs={'ContentType': 'text/html', 'ContentDisposition': 'inline'})
    return {'key': filename, 'url': f"https://f003.backblazeb2.com/file/meta-dream/{filename}"}

def upload_image(path):
    print(f"Uploading image {path} ..")

    filename = os.path.basename(path)

    result = s3.upload_file(path, config.BUCKET, filename, ExtraArgs={'ContentType': 'image/png', 'ContentDisposition': 'inline'})

    return {'key': filename, 'url': f"https://f003.backblazeb2.com/file/meta-dream/{filename}"}
