# -- coding: utf-8 --
# @Time : 2023/5/9
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
ref https://www.modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/summary
"""
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from modelscope.pipelines import pipeline

p = pipeline(
    task="document-segmentation",
    model='damo/nlp_bert_document-segmentation_chinese-base',
    device="cuda")


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str, use_document_segmentation: bool = False) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        if use_document_segmentation:
            result = p(documents=text)
            sent_list = [i for i in result["text"].split("\n\t") if i]
        else:
            sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
            sent_list = []
            for ele in sent_sep_pattern.split(text):
                if sent_sep_pattern.match(ele) and sent_list:
                    sent_list[-1] += ele
                elif ele:
                    sent_list.append(ele)
        return sent_list
