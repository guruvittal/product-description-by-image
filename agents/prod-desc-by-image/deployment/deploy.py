# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deployment script for prod descriptor agent."""

# Add the project root to sys.path
# Assumes deploy.py is in 'deployment' and project root is one level up
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import vertexai
from absl import app, flags
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
#from prod_descriptor.shared_libraries import constants
from prod_desc_by_image.agent import root_agent

"""Defines constants."""

import os
import dotenv

# Get the directory of the current script (constants.py)
basedir = os.path.abspath(os.path.dirname(__file__))
# Construct the full path to the .env file
dotenv_path = os.path.join(basedir, '.env')
print("In constants: dotenv_path:", dotenv_path)

# Load the .env file using its explicit path
# verbose=True can help debug if it finds/loads the file
dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=True)


AGENT_NAME = "product_descriptor" 
DESCRIPTION = "A helpful assistant for product description."
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "EMPTY")
bucket = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", "EMPTY")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")
DATASET_ID = os.getenv("DATASET_ID", "products_data_agent")
TABLE_ID = os.getenv("TABLE_ID", "shoe_items")
DISABLE_WEB_DRIVER = int(os.getenv("DISABLE_WEB_DRIVER", 0))
WHL_FILE_NAME = os.getenv("ADK_WHL_FILE", "")
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "agentspace_guru")

print("In constants: Staging bucket:", STAGING_BUCKET)
                                                                                                                                                        

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")
flags.DEFINE_bool("create", False, "Create a new agent.")
flags.DEFINE_bool("delete", False, "Delete an existing agent.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete"])


def create() -> None:
    adk_app = AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    extra_packages = ["./prod_desc_by_image"]

    remote_agent = agent_engines.create(
        adk_app,
        requirements=[
            "google-adk",
            "requests",
            "python-dotenv",
            "google-genai",
        ],
        extra_packages=extra_packages,
    )
    print(f"Created remote agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def main(argv: list[str]) -> None:
    project_id = FLAGS.project_id if FLAGS.project_id else PROJECT
    location = FLAGS.location if FLAGS.location else LOCATION
    bucket = FLAGS.bucket if FLAGS.bucket else STAGING_BUCKET
    print("bucket:", bucket, "FLAGS.BUCKET:", FLAGS.bucket, "Staging bucket:", STAGING_BUCKET)

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"BUCKET: {bucket}")

    if not project_id:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not location:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not bucket:
        print(
            " No bucket.Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET"
        )
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if FLAGS.create:
        create()
    elif FLAGS.delete:
        if not FLAGS.resource_id:
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    else:
        print("Unknown command")


if __name__ == "__main__":
    app.run(main)
