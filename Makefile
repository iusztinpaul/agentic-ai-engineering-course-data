compress-configs:
	cd data && rm -f configs.zip && zip -r configs.zip configs

compress-inputs:
	cd data && rm -f inputs.zip && zip -r inputs.zip inputs

compress-outputs:
	cd data && rm -f outputs.zip && zip -r outputs.zip outputs

compress-all: compress-configs compress-inputs compress-outputs
