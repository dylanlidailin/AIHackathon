# Cross-Modal Injection Probes / 跨模态注入探针

> **Do text-trained prompt-injection probes detect the same injections when they arrive as images?**
>
> **在文本上训练的提示注入探针，能否识别以图像形式出现的同一个注入？**

[English](#english) · [简体中文](#简体中文)

Status: **first full run complete** (Qwen3.5-2B, 173 + 58 LLMail pairs). Outcome: *shared direction, shifted threshold*. See [`results/qwen35_2b_v1/headline.md`](results/qwen35_2b_v1/headline.md) for the results, [`docs/quickstart.md`](docs/quickstart.md) for the HPC workflow, and [`constitution.md`](constitution.md) for the project rules.

---

## English

### TL;DR

Prompt-injection research and defenses are almost entirely text-based. But agents increasingly read **images**: screenshots, scanned documents, web pages rendered as pixels. This project asks a simple question with a cheap experiment:

> If we train a linear probe to detect prompt injections from a vision-language model's (VLM's) internal activations **using text inputs only**, does that probe still detect the **same injections when they are delivered as images**, without retraining?

The answer tells us whether a model represents "an injection is happening" the same way across modalities, and therefore whether text-trained detectors can protect multimodal agents for free, or leave a visual blind spot.

### Background

**Indirect prompt injection.** An attacker plants instructions inside content an LLM agent reads (an email, a web page, a tool output). The goal is to make the agent follow the attacker's instructions instead of the user's. This is different from a jailbreak: the injected task can be harmless (e.g. "star this repo"); the attack is that the user never asked for it.

**The modality gap (Pictionary, Zhang et al., 2026).** When the same instruction is given as text and as an image of text, VLMs follow the text version far more often (2× to 300× depending on the model). The authors trace this to text-centric instruction tuning and turn it into a defense: render untrusted text as an image before the model reads it. Two limits matter here: the defense does nothing when untrusted content is *already* an image (screenshots, GUI agents), and fine-tuning on benign image instructions erodes the gap.

**Probing within a modality (ARGUS, Lu et al., 2025).** Linear probes on VLM activations separate "following the user" from "following the injection" with near-100% accuracy, but probes were trained and tested **within** each modality.

**The open question.** Nobody has tested whether an injection signal learned from *text* transfers to *images* in the same model. That is what this repo measures.

### Research questions

| ID | Question |
|---|---|
| RQ1 | Behavioral: does our injection set reproduce the modality gap (attack success text > image) on a small VLM? |
| RQ2 | Does an **"injection present"** probe trained on text activations detect image-delivered injections? |
| RQ3 | Does a **"will obey"** probe trained on text activations predict whether the model obeys image-delivered injections? |
| RQ4 | Mechanism: if the model *registers* image injections ("present" transfers) but rarely *commits* to them ("obey" does not), that is an internal account of the modality gap. |

### Experimental design

**One model, four input conditions.** The user's task is always plain text. Only the *content* the model reads changes:

| | Content as text | Content rendered as an image |
|---|---|---|
| **Clean** | (a) clean text | (c) clean content, rendered |
| **Injected** | (b) injection in text | (d) injection, rendered |

Condition (c) matters: because the clean baseline for the image side is *also* an image, a probe cannot succeed just by detecting "there is an image".

**Three probes per layer** (logistic regression on the last prompt token's hidden state):

| Probe | Train on | Test on | Role |
|---|---|---|---|
| P1 Text sanity | text (a vs b) | held-out text | Confirms the signal exists |
| P2 **Cross-modal** | text (a vs b) | images (c vs d) | **Main question** |
| P3 Image control | images (c vs d) | held-out images | Interprets P2 |

**Controls that make a negative result believable**

- **OCR check:** before probing, confirm the model can actually transcribe the rendered images. If it can't read them, a probe failure says nothing about representations.
- **AUROC as the primary metric:** a probe can point in the right direction but have the wrong threshold on a new modality. AUROC ignores the threshold; accuracy is secondary.
- **Per-modality centering:** a reported variant that subtracts each modality's mean clean activation.
- **Grouped cross-validation by pair ID:** text and image versions of the same item never sit on opposite sides of a train/test split.
- **Separate metrics:** "target phrase emitted" and "degenerate repetition" are always reported separately.

### How to read the results

| Outcome | Interpretation |
|---|---|
| P2 works | The model encodes injections similarly across modalities; a text-trained detector covers images. |
| P2 fails, P3 works | The signal exists for images but looks different; text-only detectors leave a visual blind spot. |
| P2 and P3 both fail (OCR passes) | The model reads image instructions but does not represent them as instructions: a mechanistic view of the modality gap. |
| High AUROC, low accuracy | Shared direction, different threshold; one detector works with per-modality calibration. |
| OCR check fails | Capability limit, not a security finding; rerun on a larger model. |

All claims are scoped to the models, renderer, and data tested.

### Models

- **Primary:** Qwen3.5-2B (natively multimodal, early fusion)
- **Fast check:** Qwen3.5-0.8B
- **Fallback:** Qwen2.5-VL-3B (if the toolchain does not yet support Qwen3.5)

### Out of scope

- Jailbreaking image-*generation* models (diffusion); this project is about models that *read* images.
- Pixel-optimized adversarial images (future work).
- Any deployed or third-party system. All experiments run offline on open weights with benign target phrases.

### Repository layout

```
.
├── README.md
├── constitution.md                     # project rules
├── docs/
│   ├── quickstart.md                   # how to run on UVA HPC
│   └── papers/                         # reference papers
├── configs/                            # one YAML per model
├── src/xmprobe/                        # pipeline code (CLI stages)
├── scripts/                            # data conversion, toolchain check, post-hoc analyses
├── slurm/                              # job scripts
├── results/                            # committed summaries per run
└── tests/
```

### Running

See [`docs/quickstart.md`](docs/quickstart.md) for the step-by-step HPC workflow (environment, model download, smoke test, full run, collecting results).

### Related work

- Abdelnabi et al. *LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge* (2025). [arXiv:2506.09956](https://arxiv.org/abs/2506.09956). Source of the injection pairs (`microsoft/llmail-inject-challenge`, MIT).
- Zhang, Baroian, van Rijn, Shafran, Tramèr. *Render Before Reading: Visual Rendering as a Prompt Injection Defense* (2026). [arXiv:2609.36121](https://arxiv.org/abs/2609.36121)
- Lu et al. *ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior* (2025). [arXiv:2512.05745](https://arxiv.org/abs/2512.05745)
- Özdinçer, Simko, Schölkopf, Jin. *Improved Robustness against Indirect Prompt Injection Can Be Built into LLMs* (COLM 2026). [GitHub](https://github.com/memo-ozdincer/odile)
- Cao et al. *VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents* (ICLR 2026). [arXiv:2506.02456](https://arxiv.org/abs/2506.02456)
- Du et al. *SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents* (2026). [arXiv:2604.25562](https://arxiv.org/abs/2604.25562)

### Ethics

This project studies detection of prompt injection to support defenses. Injected instructions use benign target phrases, experiments run on local open-weight models, and no attack is run against deployed systems.

---

## 简体中文

### 一句话概括

目前关于提示注入（prompt injection）的研究和防御几乎都基于文本。但智能体（agent）越来越多地在"读图"：截图、扫描文件、以像素形式渲染的网页。本项目用一个低成本的实验回答一个简单的问题：

> 如果我们**只用文本输入**，在视觉语言模型（VLM）的内部激活上训练一个检测提示注入的线性探针（linear probe），那么当**同一个注入以图像形式出现**时，这个探针在不重新训练的情况下还能识别出来吗？

这个答案能告诉我们：模型对"正在发生注入"的内部表示是否跨模态一致；进而说明基于文本训练的检测器能否"免费"保护多模态智能体，还是会留下一个视觉盲区。

### 背景

**间接提示注入。** 攻击者把指令藏在智能体会读取的内容里（邮件、网页、工具返回结果），目的是让智能体执行攻击者的指令而不是用户的指令。这和越狱（jailbreak）不同：被注入的任务本身可以完全无害（例如"给这个仓库点星"），问题在于用户从未要求过它。

**模态差距（Pictionary，Zhang 等，2026）。** 同一条指令分别以文本和"文字图片"的形式给出时，VLM 遵循文本版本的概率远高于图片版本（因模型不同，相差 2 到 300 倍）。作者认为这源于以文本为主的指令微调，并据此提出防御：在模型读取之前，把不可信文本渲染成图片。有两点局限与本项目直接相关：当不可信内容**本来就是图片**（截图、GUI 智能体）时，该防御无效；并且在良性图像指令上微调会削弱这一差距。

**模态内探针（ARGUS，Lu 等，2025）。** 在 VLM 激活上训练的线性探针可以以接近 100% 的准确率区分"遵循用户"和"遵循注入"，但探针是在**每个模态内部**分别训练和测试的。

**尚未回答的问题。** 还没有人检验过：在同一个模型中，从**文本**学到的注入信号能否迁移到**图像**。这正是本仓库要测量的内容。

### 研究问题

| 编号 | 问题 |
|---|---|
| RQ1 | 行为层面：我们的注入数据能否在小型 VLM 上复现模态差距（文本攻击成功率 > 图像）？ |
| RQ2 | 在文本激活上训练的**"存在注入"**探针，能否识别以图像形式出现的注入？ |
| RQ3 | 在文本激活上训练的**"将会遵循"**探针，能否预测模型是否会遵循图像形式的注入？ |
| RQ4 | 机制层面：如果模型能*察觉*图像注入（"存在"探针可迁移），却很少*执行*（"遵循"探针不可迁移），这就从内部解释了模态差距。 |

### 实验设计

**同一个模型，四种输入条件。** 用户任务始终是纯文本，只改变模型读取的*内容*：

| | 内容为文本 | 内容渲染为图片 |
|---|---|---|
| **干净** | (a) 干净文本 | (c) 干净内容，渲染为图片 |
| **含注入** | (b) 文本中含注入 | (d) 含注入内容，渲染为图片 |

条件 (c) 很关键：图像一侧的干净对照组*同样是图片*，因此探针不可能仅凭"输入里有图片"就取得好成绩。

**每一层训练三个探针**（在最后一个提示 token 的隐藏状态上做逻辑回归）：

| 探针 | 训练数据 | 测试数据 | 作用 |
|---|---|---|---|
| P1 文本自检 | 文本（a 对 b） | 留出的文本 | 确认信号存在 |
| P2 **跨模态** | 文本（a 对 b） | 图像（c 对 d） | **核心问题** |
| P3 图像对照 | 图像（c 对 d） | 留出的图像 | 用于解释 P2 的结果 |

**让阴性结果可信的对照措施**

- **OCR 检查：** 在训练探针之前，先确认模型确实能转录渲染出的图片。如果模型读不懂图片，探针失败就说明不了任何关于内部表示的问题。
- **以 AUROC 作为主要指标：** 探针在新模态上可能方向正确但阈值偏移。AUROC 与阈值无关；准确率作为次要指标。
- **按模态中心化：** 额外报告一个变体，减去每个模态干净样本的平均激活。
- **按样本对 ID 分组的交叉验证：** 同一条数据的文本版本和图像版本绝不会被拆分到训练集和测试集的两侧。
- **指标分开报告：** "输出了目标短语"与"退化为重复输出"始终分开统计。

### 如何解读结果

| 结果 | 解读 |
|---|---|
| P2 有效 | 模型对注入的编码跨模态相似；文本训练的检测器可以覆盖图像。 |
| P2 失败，P3 有效 | 图像注入的信号存在但形式不同；纯文本检测器会留下视觉盲区。 |
| P2 和 P3 都失败（OCR 通过） | 模型能读懂图中的指令，但内部并未把它当作指令：从机制上解释了模态差距。 |
| AUROC 高但准确率低 | 方向一致、阈值不同；按模态校准后可用同一个检测器。 |
| OCR 检查失败 | 属于能力限制，而非安全结论；需换更大的模型重跑。 |

所有结论仅限于所测试的模型、渲染方式和数据。

### 模型

- **主要模型：** Qwen3.5-2B（原生多模态，早期融合）
- **快速验证：** Qwen3.5-0.8B
- **备选：** Qwen2.5-VL-3B（若工具链尚不支持 Qwen3.5）

### 不在本项目范围内

- 对图像*生成*模型（扩散模型）的越狱；本项目研究的是*读取*图像的模型。
- 像素级优化的对抗图像（留作后续工作）。
- 任何已部署或第三方系统。所有实验均在本地开源权重上离线运行，并使用无害的目标短语。

### 仓库结构

```
.
├── README.md
├── constitution.md                     # 项目规则
├── docs/
│   ├── quickstart.md                   # 如何在 UVA HPC 上运行
│   └── papers/                         # 参考论文
├── configs/                            # 每个模型一个 YAML
├── src/xmprobe/                        # 流水线代码（命令行各阶段）
├── scripts/                            # 数据转换、工具链检查、事后分析
├── slurm/                              # 作业脚本
├── results/                            # 每次运行提交的结果摘要
└── tests/
```

### 运行

在 HPC 上的分步流程（环境配置、模型下载、冒烟测试、完整运行、结果收集）见 [`docs/quickstart.md`](docs/quickstart.md)。

### 相关工作

- Abdelnabi 等. *LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge*（2025）. [arXiv:2506.09956](https://arxiv.org/abs/2506.09956)。注入样本对的数据来源（`microsoft/llmail-inject-challenge`，MIT 许可）。
- Zhang, Baroian, van Rijn, Shafran, Tramèr. *Render Before Reading: Visual Rendering as a Prompt Injection Defense*（2026）. [arXiv:2609.36121](https://arxiv.org/abs/2609.36121)
- Lu 等. *ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior*（2025）. [arXiv:2512.05745](https://arxiv.org/abs/2512.05745)
- Özdinçer, Simko, Schölkopf, Jin. *Improved Robustness against Indirect Prompt Injection Can Be Built into LLMs*（COLM 2026）. [GitHub](https://github.com/memo-ozdincer/odile)
- Cao 等. *VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents*（ICLR 2026）. [arXiv:2506.02456](https://arxiv.org/abs/2506.02456)
- Du 等. *SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents*（2026）. [arXiv:2604.25562](https://arxiv.org/abs/2604.25562)

### 伦理说明

本项目研究提示注入的检测，目的是支持防御。注入指令使用无害的目标短语，实验在本地开源权重模型上运行，不对任何已部署系统发起攻击。
