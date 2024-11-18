# base info
MODEL           		 =   CNN
LOSS             		 =   PL
LEARNING_LATE   		 =   1e-5
EPOCH                =   5

# data info
LABELED_NUM			     =	 1000
UNLABEL_NUM          =   1000
BATCH_SIZE           =   256

# Loss
THRESHOLD1           =   10
THRESHOLD2           =   70
DELTA1_INIT          =   0.0
DELTA2_INIT          =   0.0
DELTA1               =   3.0
DELTA2               =   1.0

# MLflow
EXPERIMENT_NAME      =   Master-thesis
TAG                  =   1.0.0

# GPU number
GPU_NUM         		 =   3 4 5

ssl:
	python src/main.py \
		--mlflow_experiment_name ${EXPERIMENT_NAME} \
		--mlflow_tag ${TAG} \
		--labeled_num ${LABELED_NUM} \
		--batch_size ${BATCH_SIZE} \
		--model ${MODEL} \
		--loss CCE \
		--learning_rate ${LEARNING_LATE} \
		--epochs ${EPOCH} \
		--threshold1 ${THRESHOLD1} \
		--threshold2 ${THRESHOLD2} \
		--delta1_init ${DELTA1_INIT} \
		--delta1 ${DELTA1} \