# Copyright (c) Microsoft Corporation and the European Centre for Medium-Range Weather Forecasts.
# Licensed under the MIT License.

import contextlib
import logging
import os
import subprocess
import traceback

import hydra

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path='../azure_configs', config_name='config')
def main(config):
    for file in config.data.files:
        scratch_file = os.path.join(config.data.env.scratch_directory, file)
        logger.info(f"reading from file {file}")
        cmd = [
            "s3cmd",
            "-r",
            "get",
            f"{config.data.env.bucket}/{file}",
            config.data.env.scratch_directory,
            "--skip-existing"
        ]
        logger.info(cmd)
        if not config.dry_run:
            try:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    subprocess.check_output(cmd)
            except subprocess.CalledProcessError:
                logger.error(f"failed to retrieve data for variable {file}")
                traceback.print_exc()
                continue

        logger.info("upload to Azure")
        url = f"{config.connection.blob_url}/{config.connection.blob_path}?{config.connection.sas_token}"
        cmd = [
            "azcopy",
            "copy",
            scratch_file,
            url,
            "--recursive",
            # "--overwrite=false"
        ]
        logger.info(cmd)
        if not config.dry_run:
            try:
                subprocess.check_output(cmd)
            except subprocess.CalledProcessError:
                logger.error(f"failed to upload data for variable {file}")
                traceback.print_exc()
                continue

        logger.info(f"removing scratch files from {scratch_file}")
        cmd = f"rm -rf {scratch_file}"
        logger.info(cmd)
        if not config.dry_run:
            os.system(cmd)

        if config.dry_run:
            logger.info("completed dry run. Execute this command with override 'dry_run=false'")


if __name__ == '__main__':
    main()
