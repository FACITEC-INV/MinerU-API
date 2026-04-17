from vllm import LLM
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor  # if vllm>=0.10.1
import pandas as pd
from ci_parser import CIParser

llm = LLM(
    model="opendatalab/MinerU2.5-2509-1.2B",
    logits_processors=[MinerULogitsProcessor]  # if vllm>=0.10.1
)

client = MinerUClient(
    backend="vllm-engine",
    vllm_llm=llm
)

image = Image.open("input/image.jpg")
extracted_blocks = client.two_step_extract(image)

df = pd.DataFrame(extracted_blocks)
parser = CIParser()

# Filtrar solo los que tienen contenido de texto
textos = df[df['content'].notnull()]['content']

resultado = "\n".join(textos)

print(resultado)
print(parser.parse(resultado))