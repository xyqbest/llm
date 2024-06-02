#!/usr/bin/env python
# coding: utf-8
# SuperICL预测分类任务
import os
import re
import time
import random

import argparse
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
import xgboost as xgb
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report

# from utils import gpt3_complete
# from templates import get_input_template, get_plugin_template


def seed_everything(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def convert_label(label, label_list):
    if label.startswith("LABEL_"):
        return label_list[int(label.split("_")[-1])]
    else:
        return label.lower()


if __name__ == "__main__":
    # =========================================
    # 超参数设置开始
    # =========================================
    # os.chdir('./insurance')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    seed_everything(seed=666)

    # 检查CUDA是否可用，并据此设置设备
    nvmlInit()
    tmp = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(_)).used for _ in range(torch.cuda.device_count())]
    gpu_id = tmp.index(min(tmp))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"This py file run on the device: 【{device}】.")

    model_name = 'multimodalmultimodal'
    model_path = f"./results/model_{model_name}_parameters.pth"
    print(f'the model is 【{model_name}】')
    # =========================================
    # 超参数设置结束
    # =========================================

    # ==========================1. 加载输入数据==========================
    # 加载数据
    df = pd.read_csv('./data/insurance_claims.csv')
    # 将分类变量 'fraud_reported' 转换为数值型
    df['fraud_reported_01'] = df['fraud_reported'].map({'N': 0, 'Y': 1})
    # 删除不需要的列
    df.drop(columns='_c39', inplace=True)
    # 将所有对象类型的列转换为类别类型
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'fraud_reported':
            df[col] = df[col].astype('category')

    # ==========================2. 加载训练好的小模型==========================
    # 定义特征 X 和目标变量 y
    # X = df.drop(columns=['fraud_reported', 'policy_bind_date', ])
    X = df.drop(columns=['fraud_reported', 'fraud_reported_01', ])
    y = df['fraud_reported_01']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    # 使用 XGBoost 训练模型
    params = {
        'n_estimators': 30,
        # 'learning_rate': 0.1,
        # 'booster': 'gbtree',
        # 'booster': 'dart',
        # 'max_depth': 12,
        # 'min_child_weight': 1,
        # 'gamma': 0.1,
        # 'subsample': .9,
        # 'colsample_bytree': .9,
        # 'colsample_bynode': .9,
        # 'objective': 'binary:logistic',
        # 'tree_method': 'hist',
        # 'tree_method': 'exact',
        # 'device': "cpu",
        # 'device': "cuda",
        # 'enable_categorical': True,
        # "max_cat_to_onehot": 60,
        # 'random_state': 666,
    }
    xgb_cls = xgb.XGBClassifier(**params, enable_categorical=True, random_state=666, ).fit(X_train, y_train)
    # 预测测试集
    y_pred = np.round(xgb_cls.predict(X_test))
    y_pred_proba = xgb_cls.predict_proba(X_test).max(axis=-1)

    # ==========================3. 基于小模型和输入数据，构造input模板==========================
    # 加载QWEN 110B开源大模型
    device = "cuda"  # the device to load the model onto
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-110B-Chat-GPTQ-Int4", cache_dir='/data2/xyq/hf_model', )
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-110B-Chat-GPTQ-Int4", torch_dtype="auto",
                                                 device_map="auto",
                                                 cache_dir='/data2/xyq/hf_model', resume_download=True)

    # prompt结构Input + Predicted Label + Confidence + Ground Truth
    # 构造Constructed Context，基于3条训练数据
    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    list_pred_xgb, list_pred_llm = [], []
    for index_test, record_test in tqdm(data_test.iterrows(), colour='green'):
    # for index_test, record_test in tqdm(data_test.head(3).iterrows()):
        # break
        # 构造训练数据示例
        train_sample = data_train.sample(n=3).reset_index(drop=True)
        xgb_pred = xgb_cls.predict(train_sample[X_train.columns])
        xgb_pred_proba = xgb_cls.predict_proba(train_sample[X_train.columns]).max(axis=-1)
        Constructed_Context = f"请你根据一些输入指标，进行车辆出险欺诈检测。我会给出几个例子，请你学习这些例子后回答。\n\n"
        for index_train, record_train in train_sample.iterrows():
            Constructed_Context += f"例{index_train + 1}：\n【输入变量】："
            for k, v in record_train.items():
                if k == 'fraud_reported_01':
                    continue
                Constructed_Context += f"{k}: {v}, "
            Constructed_Context += f"\n【小模型预测标签】：{xgb_pred[index_train]}\n" \
                                   f"【置信度】：{xgb_pred_proba[index_train]}\n" \
                                   f"【真实值】：{record_train['fraud_reported_01']}\n\n"
        # Constructed_Context += f"根据上述示例，请你预测一下记录的真实值，你预测的真实值写在'【真实值】：'后。请严格遵守输出格式要求。\n"
        Constructed_Context += f"根据上述示例，结合你的计算，请你预测以下记录的真实值。你的回答以'【真实值】：'开头，然后给出你的预测值。请严格遵守输出格式要求。\n"
        # 加入测试集数据
        record_test = pd.DataFrame([record_test.values], columns=record_test.index)
        for col in record_test.columns:
            # col='auto_make'
            if record_test[col].dtypes == object:
                record_test[col] = pd.Categorical(record_test[col], categories=X_train[col].cat.categories)

        xgb_pred = xgb_cls.predict(record_test[X_train.columns])
        xgb_pred_proba = xgb_cls.predict_proba(record_test[X_train.columns]).max(axis=-1)
        list_pred_xgb.append((xgb_pred.item(), xgb_pred_proba.item()))

        Constructed_Context += f"\n【输入变量】："
        for k, v in record_test.iloc[0].items():
            if k == 'fraud_reported_01':
                continue
            Constructed_Context += f"{k}: {v}, "
        Constructed_Context += f"\n【小模型预测标签】：{xgb_pred.item()}\n【置信度】：{xgb_pred_proba.item()}\n"
        # print(Constructed_Context)
        # 大模型预测
        prompt = Constructed_Context
        messages = [{"role": "system", "content": "你是车险领域的风险评估专家."}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=32)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        list_pred_llm.append(response)

    list_pred_llm2 = []
    for element in list_pred_llm:
        match = re.search(r'【真实值】：(\d+)', element)
        if match:
            list_pred_llm2.append(int(match.group(1)))
        else:
            list_pred_llm2.append('illegal output')
            print(element)

    list_pred_xgb2 = [_[0] for _ in list_pred_xgb]

    list_pred_true3,list_pred_xgb3,list_pred_llm3 = [],[],[]
    count = 0
    for _true,_xgb,_llm in zip(y_test.tolist(), list_pred_xgb2, list_pred_llm2):
        if _llm == 'illegal output':
            count +=1
            print(f'存在{count}个illegal output')
            continue
        list_pred_true3.append(_true)
        list_pred_xgb3.append(_xgb)
        list_pred_llm3.append(_llm)

    print("=" * 60)
    print("【XGBoost】分类结果评估:\n")
    res_clf = classification_report(list_pred_true3, list_pred_xgb3, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)
    print("【QWen 110B】分类结果评估:\n")
    res_clf = classification_report(list_pred_true3, list_pred_llm3, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)

    # ==========================4. 调用大模型给出输出==========================

    # # 中文测试
    # prompt = "解释：盖将自其变者而观之，则天地曾不能以一瞬；自其不变者而观之，则物与我皆无尽也，而又何羡乎？"
    # messages = [{"role": "system", "content": "你是中文诗词赋专家."}, {"role": "user", "content": prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=32)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
    #                  zip(model_inputs.input_ids, generated_ids)]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # prompt = "解释：盖将自其变者而观之，则天地曾不能以一瞬；自其不变者而观之，则物与我皆无尽也，而又何羡乎？"
    # messages = [{"role": "system", "content": "你是车险领域的风险评估专家."}, {"role": "user", "content": prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=32)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
    #                  zip(model_inputs.input_ids, generated_ids)]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #
    # # ==========================5. 评估模型效果==========================
    # # 计算评估指标
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print("【XGBoost】分类结果矩阵:")
    # print(conf_matrix)
    # print("=" * 60)
    # res_clf = classification_report(y_test, y_pred, digits=4, zero_division=0, )
    # print(res_clf)
    # print("=" * 60)
    #
    # # ==========================Appendix. 其它==========================
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="RoBERTa-Large",
    #     help="Name of model for prompts",
    # )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     choices=["mnli-m", "mnli-mm", "sst2", "qnli", "mrpc", "qqp", "cola", "rte"],
    #     help="Dataset to test on",
    # )
    # parser.add_argument(
    #     "--num_examples", type=int, default=32, help="Number of in-context examples"
    # )
    # parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument(
    #     "--run_icl", action="store_true", default=True, help="Run ICL baseline"
    # )
    # parser.add_argument(
    #     "--run_plugin_model",
    #     action="store_true",
    #     default=True,
    #     help="Run plugin model baseline",
    # )
    # parser.add_argument(
    #     "--run_supericl", action="store_true", default=True, help="Run SuperICL"
    # )
    # parser.add_argument(
    #     "--sleep_time", type=float, default=0.5, help="Sleep time between GPT API calls"
    # )
    # parser.add_argument(
    #     "--explanation", action="store_true", default=False, help="Run with explanation"
    # )
    # args = parser.parse_args()
    #
    # # 设置随机种子
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    #
    # plugin_model = transformers.pipeline("text-classification", model=args.model_path)
    # print(f"Loaded model {args.model_path} with name {args.model_name}")
    # print(f"Testing on dataset: {args.dataset}")
    #
    # dataset_name = args.dataset.split("-")[0]
    # dataset = datasets.load_dataset("glue", dataset_name)
    # label_list = dataset["train"].features["label"].names
    #
    # train = dataset["train"].shuffle().select(range(args.num_examples))
    # test = (
    #     dataset["validation"]
    #     if not args.dataset.startswith("mnli")
    #     else dataset[
    #         "validation" + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
    #         ]
    # )
    #
    # if args.run_icl:
    #     in_context_prompt = ""
    #     for example in train:
    #         in_context_prompt += f"{get_input_template(example, dataset_name)}\nLabel: {label_list[example['label']]}\n\n"
    #
    #     icl_predictions = []
    #     icl_ground_truth = []
    #     for example in tqdm(test):
    #         valid_prompt = (
    #                 in_context_prompt
    #                 + f"{get_input_template(example, dataset_name)}\nLabel: "
    #         )
    #         response = gpt3_complete(
    #             engine="text-davinci-003",
    #             prompt=valid_prompt,
    #             temperature=1,
    #             max_tokens=10,
    #             top_p=0.5,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #             best_of=1,
    #             stop=None,
    #         )
    #         time.sleep(args.sleep_time)
    #         icl_predictions.append(response["choices"][0]["text"].strip())
    #         icl_ground_truth.append(label_list[example["label"]])
    #
    #     if dataset_name == "cola":
    #         print(
    #             f"ICL Matthews Corr: {matthews_corrcoef(icl_predictions, icl_ground_truth)}"
    #         )
    #     else:
    #         print(f"ICL Accuracy: {accuracy_score(icl_predictions, icl_ground_truth)}")
    #
    # if args.run_plugin_model:
    #     plugin_model_predictions = []
    #     plugin_model_ground_truth = []
    #     for example in tqdm(test):
    #         plugin_model_label = convert_label(
    #             plugin_model(get_plugin_template(example, dataset_name))[0]["label"],
    #             label_list,
    #         )
    #         plugin_model_predictions.append(plugin_model_label)
    #         plugin_model_ground_truth.append(label_list[example["label"]])
    #
    #     if dataset_name == "cola":
    #         print(
    #             f"Plugin Model Matthews Corr: {matthews_corrcoef(plugin_model_predictions, plugin_model_ground_truth)}"
    #         )
    #     else:
    #         print(
    #             f"Plugin Model Accuracy: {accuracy_score(plugin_model_predictions, plugin_model_ground_truth)}"
    #         )
    #
    # if args.run_supericl:
    #     in_context_supericl_prompt = ""
    #     for example in train:
    #         plugin_input = get_plugin_template(example, dataset_name)
    #         plugin_model_result = plugin_model(plugin_input)[0]
    #         plugin_model_label = convert_label(plugin_model_result["label"], label_list)
    #         plugin_model_confidence = round(plugin_model_result["score"], 2)
    #         in_context_supericl_prompt += f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: {label_list[example['label']]}\n\n"
    #
    #     supericl_predictions = []
    #     supericl_ground_truth = []
    #     for example in tqdm(test):
    #         plugin_input = get_plugin_template(example, dataset_name)
    #         plugin_model_result = plugin_model(plugin_input)[0]
    #         plugin_model_label = convert_label(plugin_model_result["label"], label_list)
    #         plugin_model_confidence = round(plugin_model_result["score"], 2)
    #         valid_prompt = f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: "
    #         response = gpt3_complete(
    #             engine="text-davinci-003",
    #             prompt=in_context_supericl_prompt + valid_prompt,
    #             temperature=1,
    #             max_tokens=10,
    #             top_p=0.5,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #             best_of=1,
    #             stop=None,
    #         )
    #         time.sleep(args.sleep_time)
    #         supericl_prediction = response["choices"][0]["text"].strip()
    #         supericl_ground_label = label_list[example["label"]]
    #
    #         supericl_predictions.append(supericl_prediction)
    #         supericl_ground_truth.append(supericl_ground_label)
    #
    #         if args.explanation and supericl_prediction != plugin_model_label:
    #             explain_prompt = (
    #                     in_context_supericl_prompt
    #                     + valid_prompt
    #                     + "\nExplanation for overriding the prediction:"
    #             )
    #             response = gpt3_complete(
    #                 engine="text-davinci-003",
    #                 prompt=explain_prompt,
    #                 temperature=1,
    #                 max_tokens=100,
    #                 top_p=0.95,
    #                 frequency_penalty=0,
    #                 presence_penalty=0,
    #                 best_of=1,
    #                 stop=None,
    #             )
    #             print(f"\n{valid_prompt + supericl_prediction}")
    #             print(f"Explanation: {response['choices'][0]['text'].strip()}\n")
    #
    #     if dataset_name == "cola":
    #         print(
    #             f"SuperICL Matthews Corr: {matthews_corrcoef(supericl_predictions, supericl_ground_truth)}"
    #         )
    #     else:
    #         print(
    #             f"SuperICL Accuracy: {accuracy_score(supericl_predictions, supericl_ground_truth)}"
    #         )
