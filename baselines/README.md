# LLMxCPG - Baselines

## VulBERTa

The baseline is adopted from the implementation of the paper: VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection, which can be found [here](https://github.com/ICL-ml4csec/VulBERTa.git)


**Model weight**

- Please download the model weights [here](https://drive.google.com/file/d/1nElR1n_YMzCbGGgrJsVXn5G0juF1sJeA/view?usp=sharing)
- Then, unzip the file and put all model weight inside the following directory: `./VulBERTa/models`

**Setup**

In general, we used this version of packages when running the experiments:

 - Python 3.8.5
 - Pytorch 1.7.0
 - Transformers 4.4.1
 - Tokenizers 0.10.1
 - Libclang (any version > 12.0 should work. https://pypi.org/project/libclang/)

For an exhaustive list of all the packages, please refer to [requirements.txt](https://github.com/ICL-ml4csec/VulBERTa/blob/main/requirements.txt "requirements.txt") file.

**Running**

- To replicate the results for `VulBERTA-MLP`, execute the code in the following file: `./VulBERTa/Evaluation_VulBERTa-MLP.ipynb`
- To replicate the results for `VulBERTA-CNN`, execute the code in the following file: `./VulBERTa/Finetuning+evaluation_VulBERTa-CNN.ipynb`
- For `VulBERTA-CNN`, training the model is optional, just need to run the `TEST_ONLY` setting.

## ReGVD

The baseline is adopted from the implementation of the paper: ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection, which can be found [here](https://github.com/daiquocnguyen/GNN-ReGVD.git)

**Requirements**

- Python 3.7
- Pytorch 1.9
- Transformer 4.4

**Set up**

- Please download the datasets [here](https://drive.google.com/file/d/13XuKJAtNQEfr9tRnoaM8BgIoFAx5Cged/view?usp=sharing), extract, and put them in the `./GNN-ReGVD/dataset` directory.
- Pleas download the model weight [here](https://drive.google.com/file/d/1FFhKJBbX4iGe2TmEcSrCCmmb30FKcFET/view?usp=sharing), extract, and put them in the `./GNN-ReGVD/code/saved_models` directory.

**Running**
To replicate the result for this baseline, please go to the directory `./GNN-ReGVD` and run the following command:

```
cd code
python run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/sven.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt
```

## VulSim

The baseline is adopted from the implementation of the paper: VulSim: Leveraging Similarity of Multi-Dimensional Neighbor Embeddings for Vulnerability Detection, which can be found [here](https://github.com/SamihaShimmi/VulSim.git)

**Running**

- To run `code2vec`, please download the data for it [here](https://drive.google.com/file/d/1-zaFX1EeMLuD-wAouyhGIgh8opNFnHOu/view?usp=sharing) and put it in the directory `./VulSim/dx2021/astminer/dataset`. Then, go to the directory `./VulSim/dx2021/astminer/` please follow the instruction of `astminer` [here](https://github.com/dcoimbra/dx2021) with the data as follows: `{train: train2.jsonl, validation: valid2.jsonl, test:sven.jsonl}`.


- To run `SBERT`, following the original code of `VulSim`, please follow this instruction [here](https://towardsdatascience.com/sbert-vs-data2vec-on-text-classification-e3c35b19c949) to install the dependency. Then you can run the python file `./VulSim/extract_sbert.py` to extract the embeddings.

- To run `SBERT`, following the original code of `VulSim`, please follow this instruction [here](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) to install the dependency. Then you can run the python file `./VulSim/extract_codebert.py` to extract the embeddings.

- After gathering all embeddings from `code2vec, sbert, codebert`, please follow the instruction from the file `./VulSim/VulSim_classifier/README.md` to take the correlation and train the decision-tree classifier.