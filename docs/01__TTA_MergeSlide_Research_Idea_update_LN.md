# Ý tưởng Research: Test-Time Adaptation cho MergeSlide bằng cách cập nhật tham số selected của merged model

> **Tên đề xuất:** **MergeSlide-TTA** — Entropy-Guided Selective Parameter Adaptation for Domain-Shifted WSI Inference
> **Trạng thái:** Draft v2.5 — đã kiểm chứng thực nghiệm, cải thiện +1–2% bACC so với baseline (IND, TCP mode)
> **Dựa trên:** Phân tích trực tiếp từ MergeSlide paper + các bài báo TENT, SHOT, TPT, EATA trong project (v1.0), bổ sung CoTTA, PETAL, SwapPrompt (v2.5)

---

## 1. Tóm tắt ý tưởng

MergeSlide sau khi hoàn thành quá trình continual model merging tạo ra bộ tham số hợp nhất $\tilde{\theta}_{1:T}$, cùng với các class-aware prompt embeddings $\{E^{\mathcal{C},t}\}_{t=1}^T$ và task-level prompt embeddings $\mathcal{E}^T$. Khi gặp test WSI từ domain mới (OOD, cross-site domain shift), cơ chế TCP inference có thể suy giảm do phân phối slide embedding $Z_i = f_A(B'_i, \tilde{\theta}_{1:T})$ bị lệch so với phân phối tại thời điểm merging.

**Hướng đề xuất (v1.0):** Freeze toàn bộ merging coefficients $\lambda_{1:T}$ và prompt embeddings $\{E^{\mathcal{C},t}\}, \mathcal{E}^T$. Chỉ cập nhật **tham số affine của Layer Normalization** trong slide aggregator $f_A$, kết hợp cơ chế **confidence-filtered entropy minimization** ở cấp class và **diversity-regularized entropy** ở cấp task. Không cần source data, không vi phạm cấu trúc merging.

**Cập nhật hướng đề xuất (v2.5 — đã kiểm chứng):** Thực nghiệm cho thấy giả thuyết ban đầu ở mục 4 (domain shift chỉ nằm ở slide embedding, không ở task-level prompt) là **chưa đầy đủ**. Bottleneck chính nằm ở hai điểm bổ sung: (1) mục tiêu diversity ở cấp task-level trong thiết kế v1.0 bị áp dụng sai đối tượng (chi tiết mục 6.2.1), và (2) bản thân $\mathcal{E}^T$ (task-level prompt) — dù được xem là "source hypothesis" cố định giống SHOT — cũng cần được phép trôi nhẹ, có kiểm soát, theo hướng embedding space của domain hiện tại, thay vì giữ hoàn toàn tĩnh. v2.5 bổ sung: (a) sửa lại mục tiêu task-level thành đồng thuận (agreement) thay vì đa dạng (diversity), (b) thêm cơ chế **mean-teacher** (EMA của backbone) làm nguồn dự đoán ổn định hơn cho routing và loss, và (c) thêm cơ chế **task-prompt adaptation có neo (anchored EMA)** lấy cảm hứng từ SwapPrompt, ràng buộc bởi anchor để không phá vỡ tính chất "source hypothesis" của SHOT. Kết quả thực nghiệm: cải thiện +1–2% bACC so với baseline (IND, TCP mode), khớp với kỳ vọng ban đầu ở mục 9.

---

## 2. Kiến thức nền liên quan từ các bài báo TTA

### 2.1. Shannon Entropy Minimization — TENT (Wang et al., ICLR 2021)
TENT chỉ cập nhật tham số affine (scale $\gamma$, shift $\beta$) của Batch Normalization theo objective:

$$\mathcal{L}_{\text{TENT}} = -\sum_{c \in \mathcal{C}} f_\theta(c|x) \log f_\theta(c|x)$$

TENT không cần source data, hoạt động online. Hạn chế: dễ collapse khi batch nhỏ hoặc khi dữ liệu mất cân bằng. **Điểm kế thừa:** cập nhật normalization parameters là chiến lược nhẹ nhất, trực tiếp ứng với Layer Norm của transformer trong MergeSlide.

### 2.2. Information Maximization — SHOT (Liang et al., TPAMI 2021)
SHOT freeze classifier (source hypothesis) và chỉ update feature extractor qua loss ghép hai thành phần:

$$\mathcal{L}_{\text{IM}} = \mathcal{L}_{\text{ent}} + \beta \mathcal{L}_{\text{div}}$$

$$\mathcal{L}_{\text{ent}} = -\mathbb{E}_{x \in \mathcal{X}_t}\left[\sum_k \delta_k(f(x)) \log \delta_k(f(x))\right]$$

$$\mathcal{L}_{\text{div}} = D_{\text{KL}}\left(\hat{p} \,\Big\|\, \frac{1}{K}\mathbf{1}_K\right) - \log K, \quad \hat{p} = \mathbb{E}_{x}[\delta(f(x))]$$

$\mathcal{L}_{\text{div}}$ ngăn hiện tượng collapse (tất cả mẫu dự đoán vào một class). **Điểm kế thừa:** trong MergeSlide, prompt embeddings đóng vai trò "source hypothesis" cố định — tương tự cơ chế freeze classifier của SHOT.

> **Cập nhật v2.5 (Confirmed bằng thực nghiệm):** $\mathcal{L}_{\text{div}}$ của SHOT giả định ngầm rằng các mẫu trong một batch có **nhãn thật khác nhau**, nên phân phối dự đoán trung bình của batch *nên* gần uniform để tránh collapse-về-một-lớp. Giả định này **đúng** khi áp dụng cho class-level prediction trên các sub-bag của cùng một slide (các sub-bag có thể chứa vùng mô khác nhau về hình thái). Tuy nhiên giả định này **sai** khi áp dụng cho task-routing logits của các sub-bag thuộc cùng một slide — vì toàn bộ sub-bag của một slide chia sẻ **cùng một nhãn task thật**, nên mục tiêu đúng là các sub-bag *đồng thuận* (gần one-hot), không phải *đa dạng* (gần uniform). Đây là bug đã được xác nhận trực tiếp qua thực nghiệm (routing accuracy của task yếu nhất giảm mạnh khi tăng $n_{\text{steps}}$ dưới thiết kế v1.0, do gradient liên tục đẩy routing distribution sai hướng). Xem chi tiết cơ chế sửa ở mục 6.2.1.

### 2.3. Confidence Filtering — EATA (Niu et al., ICML 2022)
EATA thêm sample-adaptive weight $S(x)$ vào entropy loss:

$$S^{\text{ent}}(x) = \frac{1}{\exp[E(x;\Theta) - E_0]} \cdot \mathbb{I}_{\{E(x;\Theta) < E_0\}}(x)$$

Chỉ backward trên mẫu **đáng tin** (entropy thấp) và **không trùng lặp** (cosine diversity). Đồng thời dùng **Fisher regularization** chống catastrophic forgetting. **Điểm kế thừa:** Với WSI, mỗi slide chứa hàng nghìn patch — cần lọc patch đáng tin trước khi tính loss để tránh gradient nhiễu.

> **Cập nhật v2.5:** Thiết kế v1.0 lọc sub-bag đáng tin theo class **hoặc** theo task (union). Điều này cho phép một sub-bag "tự tin sai" ở một trong hai tiêu chí (ví dụ: routing task sai nhưng entropy class thấp giả tạo) vẫn lọt vào tập backward. v2.5 chuyển sang **intersection** — chỉ giữ sub-bag đáng tin ở **cả hai** tiêu chí đồng thời — bám sát tinh thần lọc chặt của EATA hơn thiết kế v1.0, giảm nguy cơ gradient bị dẫn dắt bởi sub-bag tự tin nửa vời.

### 2.4. Test-Time Prompt Tuning — TPT (Shu et al., NeurIPS 2022)
TPT tối ưu learnable text prompt tokens tại test time qua marginal entropy trên nhiều augmented views, giữ nguyên toàn bộ backbone VLM:

$$p^* = \arg\min_p H\left(\tilde{p}_p(y|X)\right), \quad \tilde{p}_p = \frac{1}{N}\sum_{n=1}^{N} p_p(y|\mathcal{A}_n(X))$$

Kết quả ablation của TPT chứng minh: **text prompt là parameter group hiệu quả nhất**, tốt hơn cả fine-tune full encoder, vì không làm méo pre-trained features. **Điểm kế thừa:** gợi ý rằng trong MergeSlide, nếu cần thêm một nhánh adaptation, learnable prompt offset nhỏ (không thay thế $E^{\mathcal{C},t}$ mà chỉ thêm delta) là hướng an toàn.

> **Cập nhật v2.5:** Ý tưởng "prompt offset nhỏ" nêu ở v1.0 (mục "Tùy chọn" trong Next steps) đã được hiện thực hoá và kiểm chứng — xem mục 2.7 (SwapPrompt) và 6.4 (Task-Prompt Embedding-Space Adaptation). Đồng thời, nguyên lý **confidence gate** của TPT (chỉ cập nhật khi mẫu đủ tin cậy) được tái sử dụng làm điều kiện kích hoạt việc cập nhật task-prompt, không phải cập nhật prompt tại mọi bước.

### 2.5. Anti-Forgetting Regularization — EATA (Fisher Regularizer)
EATA dùng Fisher information để đo mức độ quan trọng của từng weight $\omega(\theta_i)$, sau đó thêm regularizer:

$$\mathcal{R}(\tilde{\Theta}, \tilde{\Theta}^o) = \sum_{\theta_i \in \tilde{\Theta}} \omega(\theta_i)(\theta_i - \theta_i^o)^2$$

**Điểm kế thừa:** MergeSlide đã có Condition b đảm bảo $\|\tilde{\theta}_{1:t} - \theta_{\text{base}}\|_2$ bị chặn trên. Fisher regularizer tại test time là cơ chế analog để duy trì ràng buộc tương tự trong quá trình TTA.

> **Cập nhật v2.5:** Bản implement hiện tại dùng **L2-anchor regularizer đơn giản** (không ước lượng Fisher đầy đủ) cho LN params — $\mathcal{R}_{\text{forget}} = \beta \sum \|\theta_i - \theta_i^0\|_2^2$ — thay vì trọng số theo Fisher importance $\omega(\theta_i)$. Đây là lựa chọn "đơn giản hơn nhưng tương đương" mà chính rủi ro ở mục 10 (bảng gốc) đã dự tính trước ("nếu tương đương thì dùng đơn giản hơn"). Nguyên lý neo (anchor) tương tự cũng được mở rộng sang task-prompt embedding ở mục 6.4 — đây là điểm khác biệt so với thiết kế Fisher-restoration đầy đủ (dùng Fisher Information Matrix để chọn lọc param nào được phục hồi về giá trị gốc), vốn có chi phí tính toán cao hơn và được để lại như một hướng mở rộng (mục 12).

### 2.6. Mean-Teacher / Consistency Adaptation — CoTTA-inspired (Wang et al., CVPR 2022)
CoTTA duy trì một teacher model là **exponential moving average (EMA)** của student model đang được adapt, và dùng dự đoán của teacher (ổn định hơn, ít nhiễu hơn do không nhận gradient trực tiếp) làm pseudo-target cho student:

$$\theta_{\text{teacher}} \leftarrow \eta \, \theta_{\text{teacher}} + (1-\eta)\, \theta_{\text{student}}$$

**Điểm kế thừa (v2.5, mới):** Trong online continual TTA, backbone (đóng vai trò "student") liên tục nhận gradient qua nhiều slide — dự đoán tức thời của nó có thể dao động mạnh ngay sau mỗi bước cập nhật. Dùng một bản EMA ổn định hơn (teacher) để quyết định **routing** (chọn task $\hat{t}$) và **suy luận cuối cùng** giúp tách bạch vai trò "nơi nhận gradient" (student) và "nơi ra quyết định" (teacher), giảm rủi ro một bước gradient đơn lẻ (nhiễu) làm lệch routing của chính slide đang xử lý.

### 2.7. Prompt Adaptation via EMA — SwapPrompt-inspired (Ma et al., NeurIPS 2023)
SwapPrompt (unsupervised domain adaptation) cập nhật prompt embedding tại test time bằng EMA hướng về embedding của dữ liệu target, có điều kiện tin cậy (confidence gate) để tránh nhiễm nhiễu từ dự đoán không chắc chắn:

$$p_t \leftarrow \eta_p \, p_t + (1-\eta_p)\, z, \quad \text{chỉ khi } \text{margin}(z) > \delta$$

**Điểm kế thừa (v2.5, mới):** Task-level prompt $\mathcal{E}^T$ trong MergeSlide đóng vai trò "source hypothesis" cố định (giống SHOT, mục 2.2). Tuy nhiên khi domain shift đủ lớn, hypothesis cố định có thể không còn là điểm tham chiếu tối ưu cho routing. v2.5 cho phép $\mathcal{E}^T$ trôi nhẹ theo domain hiện tại — **có điều kiện tin cậy** (chỉ cập nhật khi gap giữa task dẫn đầu và task nhì đủ lớn, tránh nhiễm nhiễu từ routing không chắc chắn) và **có neo** (xem mục 6.4) để không vi phạm hoàn toàn giả định "source hypothesis cố định" của SHOT — đây là điểm dung hoà giữa hai triết lý (SHOT: giữ nguyên hypothesis vs. SwapPrompt: cho phép hypothesis thích nghi).

---

## 3. Bảng đối chiếu nguồn gốc ý tưởng

| Thành phần ý tưởng                          | Bài báo gốc liên quan                   | Phương pháp của bài báo gốc                                             | Cách chuyển sang MergeSlide                                                                                                     | Lý do phù hợp                                                                                             | Rủi ro                                                                                                  |
| ------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Freeze merging coefficients $\lambda_{1:T}$ | SHOT (Liang et al., TPAMI 2021)         | Freeze source classifier (hypothesis), chỉ update feature extractor     | Giữ nguyên $\lambda_{1:T}$ (merging structure) và prompt embeddings; chỉ update LN params                                       | Lambda được thiết kế để duy trì $\|\tilde{\theta} - \theta_{\text{base}}\|_2$; update sẽ phá cấu trúc này | LN params có thể không đủ expressive để bù domain shift lớn — **Confirmed một phần**: LN đủ cho +1–2%, không đủ để đóng hoàn toàn domain gap (xem mục 10) |
| Update Layer Normalization affine params    | TENT (Wang et al., ICLR 2021)           | Update BN affine params (scale $\gamma$, shift $\beta$) via entropy min | Thay BN bằng LN (TITAN/Transformer dùng LN), cập nhật $\gamma^l, \beta^l$ cho mỗi lớp $l$ của $f_A$                             | MergeSlide dùng Transformer aggregator (LN, không phải BN); TENT principle trực tiếp áp dụng              | Transformer có nhiều LN layers; cần xác định layer nào update                                             |
| Class-level entropy minimization            | TENT, SHOT                              | $\mathcal{L}_{\text{ent}} = -\sum_c p_c \log p_c$ trên class prediction | Tính entropy trên dot-product similarity score giữa $Z_i$ và $\{E^{\mathcal{C},t}\}$                                            | Class-aware prompts đã fixed; entropy trên similarity score có ý nghĩa trực tiếp                          | Nếu nhiều task có class giống nhau (NSCLC/ESCA đều có adenocarcinoma), entropy có thể bị lẫn            |
| ~~Task-level diversity regularization~~ **→ Task-level agreement regularization (v2.5)** | ~~SHOT ($\mathcal{L}_{\text{div}}$)~~ **→ CoTTA-inspired (consistency)** | ~~KL divergence giữa mean prediction và uniform distribution~~ **→ Jensen-Shannon Divergence giữa các sub-bag** | ~~Tính $\mathcal{L}_{\text{div}}$ trên task prediction score~~ **→ Tính JSD giữa phân phối task-routing của các sub-bag cùng slide, minimize để ép đồng thuận** | **Sửa lỗi v1.0:** các sub-bag cùng slide chia sẻ cùng nhãn task thật, nên mục tiêu đúng là đồng thuận (one-hot), không phải đa dạng (uniform) — xem mục 6.2.1 | Nếu agreement quá mạnh có thể làm routing hội tụ sớm về task sai nếu đa số sub-bag ban đầu bị routing sai |
| Confidence filtering trên patches/sub-bag           | EATA ($S^{\text{ent}}(x)$), FS-TTA      | Lọc sample có entropy thấp, loại sample entropy cao                     | Lọc sub-bag theo entropy; chỉ backward trên sub-bag đáng tin — **v2.5: đổi từ union sang intersection giữa 2 tiêu chí class + task**              | WSI có hàng nghìn patches, nhiều là background/normal tissue; gradient từ noisy patches phá adaptation; intersection lọc chặt hơn, bám sát tinh thần EATA hơn         | Ngưỡng $E_0$/top-ratio quá chặt có thể khiến quá ít sub-bag được chọn, adaptation yếu             |
| Anti-forgetting regularization              | EATA (Fisher), EWC (Kirkpatrick et al.) | Fisher regularizer $\omega(\theta_i)(\theta_i - \theta_i^o)^2$          | Regularize LN params về $\tilde{\theta}_{1:T}$ (giá trị trước TTA) — **bản hiện tại dùng L2-anchor đơn giản, không ước lượng Fisher** | Duy trì Condition b của MergeSlide: $\|\tilde{\theta}_{1:t} - \theta_{\text{base}}\|_2$ bị chặn           | Fisher đầy đủ để lại như hướng mở rộng; L2-anchor đơn giản hơn nhưng thực nghiệm cho kết quả chấp nhận được |
| **(Mới, v2.5) Mean-teacher stabilization**  | CoTTA-inspired (Wang et al., CVPR 2022) | Teacher = EMA(student), dùng cho pseudo-label/quyết định cuối          | Teacher = EMA(backbone); dùng $z_{\text{teacher}}$ cho routing task và suy luận cuối, backbone ("student") vẫn là nơi nhận gradient | Tách vai trò "nơi học" và "nơi quyết định", giảm nhiễu từ dao động gradient tức thời                      | Teacher có độ trễ (lag) so với domain shift thật sự nhanh; cần theo dõi nếu domain shift xảy ra đột ngột |
| **(Mới, v2.5) Task-prompt embedding-space adaptation** | SwapPrompt-inspired (Ma et al., NeurIPS 2023) + TPT (confidence gate) | EMA cập nhật prompt hướng về target embedding, có gate tin cậy | Cập nhật $\mathcal{E}^T_t$ hướng về $z_{\text{teacher}}$ khi routing margin đủ lớn; có thêm **anchor pull-back** về $\mathcal{E}^T_{t,\text{source}}$ | Trực tiếp giải quyết bottleneck "routing chưa tốt" quan sát được ở TCP mode — task-prompt vốn cố định có thể không còn là điểm tham chiếu tối ưu dưới domain shift | Rủi ro tự tham chiếu (echo chamber/confirmation bias): nếu vài sub-bag đầu bị routing sai nhưng tự tin, prompt có thể trôi theo hướng sai; anchor + reset theo từng task (mục 6.4) là cơ chế giảm thiểu |
| **(Mới, v2.5) Task-level margin loss**      | Lấy cảm hứng từ large-margin softmax, kết hợp với task routing | Ép gap giữa lớp dẫn đầu và lớp nhì lớn hơn ngưỡng margin | $\mathcal{L}_{\text{margin}} = \text{ReLU}(s_{\text{top2}} - s_{\text{top1}} + m)$ trên task score, không giả định routing hiện tại đúng | Bổ trợ cho agreement loss: agreement ép các sub-bag *thống nhất*, margin ép độ *sắc nét* của quyết định dẫn đầu (dù đúng hay sai) | Nếu bật đồng thời với agreement loss ở trọng số lớn có thể làm tổng gradient quá mạnh, cần ablate riêng |

---

## 4. Giả thuyết nghiên cứu

> **Giả thuyết chính (v1.0):** Khi MergeSlide gặp domain shift (cross-site OOD), sự suy giảm hiệu năng chủ yếu đến từ **sự lệch phân phối của slide embeddings $Z_i$** so với các prompt embeddings $\{E^{\mathcal{C},t}\}$ — không phải từ sai lệch trong merging coefficients $\lambda_{1:T}$. Do đó, cập nhật nhẹ các tham số affine normalization trong $f_A$ đủ để căn chỉnh lại phân phối embedding, trong khi giữ nguyên cấu trúc merging bảo toàn tri thức đa nhiệm.

**Kiểm chứng được khi:** Mô hình sau TTA có bACC cao hơn MergeSlide gốc trên OOD setting (cross-site), trong khi không suy giảm trên in-domain test set.

> **Giả thuyết bổ sung (v2.5 — Confirmed một phần bằng thực nghiệm):** Giả thuyết chính ở trên là **cần nhưng chưa đủ**. Thực nghiệm cho thấy: (1) đóng góp lớn nhất của TTA nằm ở việc **sửa mục tiêu task-level** (đổi diversity→agreement, mục 6.2.1), không chỉ đơn thuần "cập nhật LN đủ để căn chỉnh embedding" như giả thuyết gốc ngụ ý; (2) một phần đáng kể của độ lệch không chỉ nằm ở slide embedding $Z_i$ mà còn ở **độ phù hợp của task-level prompt $\mathcal{E}^T$ với domain hiện tại** — cho phép $\mathcal{E}^T$ trôi nhẹ, có neo, đóng góp thêm vào cải thiện tổng thể. Nói cách khác: bottleneck domain shift trong TCP inference có **hai nguồn**, không phải một — (a) slide embedding lệch khỏi prompt cố định (giả thuyết gốc), và (b) chính prompt cố định có thể không còn là điểm neo tối ưu cho domain hiện tại (giả thuyết bổ sung). Đánh giá mức độ bằng chứng: **Likely** cho thành phần (b) — đã quan sát cải thiện thực nghiệm nhất quán, nhưng chưa tách bạch định lượng chính xác đóng góp riêng của (a) so với (b) qua ablation đầy đủ (xem mục 9).

**Kiểm chứng được khi (v2.5):** (i) Tắt task-prompt adaptation (chỉ giữ LN + agreement loss) → so sánh bACC với bản đầy đủ; (ii) đo độ dịch chuyển $\|\mathcal{E}^T_{t} - \mathcal{E}^T_{t,\text{source}}\|_2$ cuối mỗi task — nếu dịch chuyển nhỏ mà routing vẫn cải thiện, cần tìm nguồn cải thiện khác; nếu dịch chuyển tương quan thuận với mức cải thiện, củng cố giả thuyết (b).

---

## 5. Phương pháp đề xuất: MergeSlide-TTA

### Tên: MergeSlide-TTA: Selective Normalization Adaptation with Dual-Level Entropy Guidance and Anchored Prompt-Space Adaptation *(tên mở rộng, v2.5)*

### Thành phần Frozen:
- Merging coefficients $\lambda_{1:T}$ và merged weights $\tilde{\theta}_{1:T}$ (trừ LN affine params)
- Text encoder của TITAN
- Class-aware prompt embeddings $\{E^{\mathcal{C},t}\}_{t=1}^T$
- **(v2.5) Task-level prompt gốc $\mathcal{E}^T_{\text{source}}$** — giữ vai trò điểm neo (anchor), không bị ghi đè vĩnh viễn

### Thành phần được Update:
- Tham số affine của **Layer Normalization** trong $f_A$: $\{\gamma^l, \beta^l\}_{l=1}^L$ (chỉ scale và shift, không update attention weights hay FFN weights)
- **(v2.5, mới) Task-level prompt working copy $\mathcal{E}^T$** — cập nhật có điều kiện (confidence gate) và có neo về $\mathcal{E}^T_{\text{source}}$, không update bằng gradient trực tiếp mà bằng EMA (giống SwapPrompt), tách biệt hoàn toàn khỏi cơ chế gradient-descent của LN params

### Thành phần bổ sung vai trò (không phải tham số học được):
- **(v2.5, mới) Teacher model** — bản EMA của backbone, không nhận gradient trực tiếp, dùng để routing task và ra quyết định cuối, ổn định hơn backbone đang trong quá trình adapt

### Lý do không update lambda:
$\lambda_t$ trong MergeSlide được thiết kế theo Eq. (5) để đảm bảo $\|\tilde{\theta}_{1:t} - \theta_{\text{base}}\|_2 \leq \max_i \|\theta_i\|_2$. Thay đổi $\lambda_t$ tại test time sẽ phá vỡ ràng buộc này, làm mất tính chất "không drift quá xa base model" — đây là bảo đảm chống catastrophic forgetting cốt lõi của MergeSlide.

### Lý do update LN params hợp lý hơn:
Layer Normalization chỉ có $2d$ tham số (với $d$ là hidden dim), không thay đổi kiến trúc. Các tham số $\gamma, \beta$ quyết định **phân phối feature** tại mỗi lớp nhưng không ảnh hưởng đến **hướng** của merged task vectors $\Delta\tilde{\theta}_{1:t}$. Tương tự như TENT đã chứng minh: BN/LN affine params là parameter group tối ưu cho adaptation — đủ expressive nhưng không gây catastrophic forgetting.

### Lý do cho phép task-prompt trôi có kiểm soát (v2.5, mới):
Khác với $\lambda_t$ (encode cấu trúc merging, không được phép đổi) và class-prompt $E^{\mathcal{C},t}$ (định nghĩa chính bản thân class, không được phép đổi), task-prompt $\mathcal{E}^T$ chỉ đóng vai trò **điểm tham chiếu để routing**, không định nghĩa nhãn. Cho phép nó dịch chuyển nhẹ, có gate tin cậy và có neo, không vi phạm các ràng buộc cốt lõi (Condition b của MergeSlide) vì: (1) biên độ dịch chuyển bị chặn bởi hệ số neo, (2) mỗi task được reset về nguồn khi bắt đầu xử lý task mới trong luồng test tuần tự, tránh trôi tích luỹ xuyên suốt nhiều task — đây là khác biệt so với thiết kế gốc lấy cảm hứng SwapPrompt (vốn không có bước neo/reset này).

### Đảm bảo không cần source data:
Tất cả gradients chỉ dựa trên dự đoán của model trên test WSI hiện tại. Không cần lưu trữ hay truy cập WSI training. Thông tin duy nhất cần giữ lại: (1) frozen model weights $\tilde{\theta}_{1:T}$, (2) prompt embeddings gốc (bao gồm cả $\mathcal{E}^T_{\text{source}}$ để làm neo), (3) giá trị LN params ban đầu $\{\gamma^l_0, \beta^l_0\}$ để regularize.

---

## 6. Thiết kế loss

Loss tổng quát (v2.5, mở rộng từ v1.0):

$$\mathcal{L}_{\text{TTA}} = \mathcal{L}_{\text{class}} + \alpha \mathcal{L}_{\text{task}} + \gamma_m \mathcal{L}_{\text{margin}} + \beta \mathcal{R}_{\text{forget}}$$

trong đó $\mathcal{L}_{\text{task}}$ đã được định nghĩa lại so với v1.0 (mục 6.2.1) và $\mathcal{L}_{\text{margin}}$ là thành phần mới (mục 6.3.1). Việc cập nhật task-prompt $\mathcal{E}^T$ (mục 6.4) **không** nằm trong $\mathcal{L}_{\text{TTA}}$ — đây là một cơ chế EMA riêng biệt, tách khỏi vòng lặp gradient-descent trên LN params, để tránh trộn lẫn hai cơ chế học có bản chất khác nhau (gradient-based vs. EMA-based) trong cùng một hàm mục tiêu.

### 6.1. Class-level adaptation loss $\mathcal{L}_{\text{class}}$

Lấy cảm hứng từ **SHOT** (Information Maximization) và **EATA** (confidence filtering):

$$\mathcal{L}_{\text{class}} = \underbrace{-\sum_{c=1}^{C} \tilde{p}_c \log \tilde{p}_c}_{\text{entropy min (TENT/SHOT)}} - \underbrace{\sum_{c=1}^{C} \bar{p}_c \log \bar{p}_c}_{\text{diversity (SHOT } \mathcal{L}_{\text{div}}\text{)}}$$

Trong đó:
- $\tilde{p}_c = \text{softmax}(Z_i \cdot (E^{\mathcal{C},\hat{t}})^\top)_c$: xác suất class $c$ sau khi TCP xác định task $\hat{t}$
- $\bar{p}_c = \frac{1}{|\mathcal{B}_K|}\sum_{i \in \mathcal{B}_K} \tilde{p}_c^{(i)}$: mean prediction trên batch các patch **đã được lọc** $\mathcal{B}_K$
- $\mathcal{B}_K$: tập $K$ patches có entropy thấp nhất (top-$K\%$ confident patches, theo EATA/FS-TTA):

$$\mathcal{B}_K = \left\{i : H(\tilde{p}^{(i)}) < E_0\right\}, \quad E_0 = \text{threshold}$$

**Vai trò từng thành phần:**
- Entropy minimization: ép model đưa ra dự đoán class chắc chắn hơn trên target domain.
- Diversity term (âm của marginal entropy): ngăn collapse về một class, đặc biệt quan trọng với ung thư mất cân bằng (ví dụ NSCLC vs. TGCT).

**Giữ nguyên ở v2.5** — diversity term ở cấp class là hợp lệ về mặt giả định (các sub-bag khác nhau có thể chứa mô khác nhau về hình thái, nên phân phối class không nhất thiết one-hot), khác với trường hợp task-level bên dưới.

### 6.2. Task-level adaptation loss $\mathcal{L}_{\text{task}}$ — **[ĐÃ SỬA LỖI ở v2.5]**

#### 6.2.1. Vấn đề của thiết kế v1.0

Thiết kế gốc:

$$\mathcal{L}_{\text{task}}^{\text{(v1.0)}} = -\sum_{t=1}^{T} q_t \log q_t - \lambda_{\text{div}} \cdot \sum_{t=1}^{T} \bar{q}_t \log \bar{q}_t$$

áp dụng cùng công thức diversity của SHOT (mục 2.2/6.1) cho task-routing logits. Đây là bug đã được xác nhận qua thực nghiệm: mọi sub-bag của **cùng một slide** chia sẻ **cùng một nhãn task thật** — phân phối task trung bình lý tưởng $\bar{q}_t$ phải gần **one-hot** (các sub-bag đồng thuận), không phải gần **uniform** như giả định diversity của SHOT. Áp dụng diversity term ở đây tương đương với việc chủ động huấn luyện mô hình tạo ra dự đoán **mâu thuẫn** giữa các sub-bag của cùng một slide — ngược hướng với mục tiêu routing đúng. Bằng chứng thực nghiệm: routing accuracy của task yếu nhất trong chuỗi continual **giảm** khi tăng số bước cập nhật $n_{\text{steps}}$ dưới thiết kế v1.0 — nhất quán với việc mỗi bước gradient là một bước đi sai hướng tích luỹ.

#### 6.2.2. Thiết kế đã sửa (v2.5)

Thay thế diversity term bằng **agreement term** dựa trên Jensen-Shannon Divergence, lấy cảm hứng từ nguyên lý consistency regularization của CoTTA (mục 2.6):

$$\mathcal{L}_{\text{task}}^{\text{(v2.5)}} = \underbrace{-\sum_{t=1}^{T} q_t \log q_t}_{\text{entropy min, giữ nguyên}} + \gamma_a \cdot \underbrace{\frac{1}{M}\sum_{i=1}^{M} D_{\text{KL}}\left(q^{(i)} \,\Big\|\, \bar{q}\right)}_{\mathcal{L}_{\text{agree}}\text{ (agreement, thay thế diversity)}}$$

Trong đó:
- $q^{(i)}_t = \text{softmax}(Z_i^{(i)} \cdot (E^{\mathcal{T},t})^\top)_t$: phân phối task-routing của sub-bag thứ $i$ trong $M$ sub-bag của cùng một slide
- $\bar{q} = \frac{1}{M}\sum_{i=1}^{M} q^{(i)}$: phân phối task trung bình qua $M$ sub-bag
- $\gamma_a$: trọng số agreement term

**Vai trò:** Minimize $\mathcal{L}_{\text{agree}}$ kéo các phân phối $q^{(i)}$ của $M$ sub-bag lại gần nhau (đồng thuận), đúng với ràng buộc "cùng slide → cùng task thật", thay vì đẩy chúng ra xa nhau (SHOT-style diversity, chỉ đúng khi các mẫu trong batch có nhãn khác nhau). Đây là thành phần đóng góp chính vào cải thiện +1–2% quan sát được, vì nó trực tiếp sửa bottleneck: routing sai là nguyên nhân chính khiến TCP inference kém, không phải class-boundary chưa đủ sắc nét.

**Vai trò của entropy term (giữ nguyên từ v1.0):** Task prediction accuracy là nút thắt quan trọng của TCP inference. Nếu model bị domain shift khiến task prediction sai, class prediction sau đó cũng sai. Tối ưu phần entropy trực tiếp sửa alignment giữa slide embedding và task-level prompt, phần agreement đảm bảo alignment đó nhất quán qua các sub-bag của cùng slide.

### 6.3. Anti-forgetting regularizer $\mathcal{R}_{\text{forget}}$

Lấy cảm hứng từ **EATA** (Fisher regularizer), bản hiện tại dùng dạng đơn giản hoá — L2-anchor không trọng số Fisher:

$$\mathcal{R}_{\text{forget}} = \sum_{l=1}^{L} \left[(\gamma^l - \gamma^l_0)^2 + (\beta^l - \beta^l_0)^2\right]$$

Trong đó $\gamma^l_0, \beta^l_0$ là giá trị LN params trước khi TTA (từ merged model $\tilde{\theta}_{1:T}$). Việc chuyển từ Fisher-weighted regularizer (v1.0, mục 6.3 gốc) sang L2-anchor không trọng số là lựa chọn đơn giản hoá có chủ đích — tránh phụ thuộc vào chất lượng ước lượng pseudo-Fisher từ dữ liệu test không nhãn (rủi ro đã nêu ở mục 10 bảng gốc), đổi lại bằng một regularizer kém "chọn lọc" hơn nhưng ổn định hơn.

### 6.3.1. Task-level margin loss $\mathcal{L}_{\text{margin}}$ *(mới, v2.5)*

Lấy cảm hứng từ large-margin objectives, áp dụng lên chính task-routing score (không phải trên đầu ra class):

$$\mathcal{L}_{\text{margin}} = \text{ReLU}\left(s_{\text{top2}} - s_{\text{top1}} + m\right)$$

Trong đó $s_{\text{top1}}, s_{\text{top2}}$ là điểm số (dot-product similarity) của task dẫn đầu và task đứng thứ hai theo $\mathcal{E}^T$, $m$ là margin mong muốn. Loss bằng 0 khi khoảng cách giữa hai task đã đủ lớn.

**Vai trò:** Bổ trợ cho $\mathcal{L}_{\text{agree}}$ — trong khi agreement ép các sub-bag *thống nhất với nhau* về việc chọn task nào, margin loss ép quyết định dẫn đầu (dù đúng hay sai) phải *rõ ràng, dứt khoát* (tránh vùng biên mơ hồ giữa hai task gần nhau về mặt hình thái, ví dụ các cặp cohort có đặc điểm mô học chồng lấn). Hai thành phần này độc lập về mặt mục tiêu (đồng thuận vs. độ sắc nét), có thể bật/tắt riêng để ablate.

### 6.4. Task-Prompt Embedding-Space Adaptation *(mới, v2.5 — thành phần cốt lõi giải quyết bottleneck routing)*

Đây là cơ chế **không** thuộc $\mathcal{L}_{\text{TTA}}$ — không cập nhật bằng gradient descent, mà bằng EMA có điều kiện, tách biệt khỏi vòng lặp tối ưu LN params.

**Điều kiện kích hoạt (confidence gate, TPT-inspired):** Chỉ cập nhật $\mathcal{E}^T_{\hat{t}}$ (prompt của task đang được routing) khi khoảng cách tin cậy giữa task dẫn đầu và task nhì vượt ngưỡng:

$$\text{margin} = \text{softmax}(\bar{Z} \cdot \mathcal{E}^{T\top})_{\text{top1}} - \text{softmax}(\bar{Z} \cdot \mathcal{E}^{T\top})_{\text{top2}} > \delta_{\text{margin}}$$

trong đó $\bar{Z}$ là embedding trung bình của các sub-bag trong slide hiện tại (lấy từ nguồn ổn định — teacher, mục 2.6 — không phải backbone đang giữa bước gradient).

**Cập nhật EMA có neo (anchored EMA — mở rộng so với SwapPrompt gốc):**

$$\mathcal{E}^T_{\hat{t}} \leftarrow (1-\beta_{\text{anchor}}) \cdot \Big[\eta_p \, \mathcal{E}^T_{\hat{t}} + (1-\eta_p)\, \bar{Z}\Big] \; + \; \beta_{\text{anchor}} \cdot \mathcal{E}^T_{\hat{t},\text{source}}$$

Trong đó $\eta_p$ là hệ số EMA (giống SwapPrompt gốc), và $\beta_{\text{anchor}} \in [0,1]$ là hệ số neo **mới bổ sung** — kéo prompt về giá trị nguồn sau mỗi lần cập nhật, không có trong thiết kế SwapPrompt/tta_engine gốc.

**Lý do bổ sung hệ số neo:** Cơ chế EMA thuần tuý (không neo, $\beta_{\text{anchor}}=0$) tạo ra một vòng lặp tự tham chiếu tiềm ẩn: $\mathcal{E}^T_{\hat{t}}$ dùng để routing slide hiện tại, sau đó nếu routing "tự tin", chính embedding của slide đó lại được dùng để cập nhật $\mathcal{E}^T_{\hat{t}}$ — khiến các slide tiếp theo dễ được routing vào task đó hơn (dù đúng hay sai), tạo rủi ro confirmation bias / echo-chamber tích luỹ theo thứ tự xử lý slide trong luồng test tuần tự. Hệ số neo giới hạn biên độ trôi tối đa, giữ lại lợi ích thích nghi (embedding space theo domain hiện tại) trong khi giảm rủi ro trôi không kiểm soát.

**Reset theo ranh giới task:** $\mathcal{E}^T$ được reset về $\mathcal{E}^T_{\text{source}}$ khi bắt đầu xử lý một task mới trong luồng test tuần tự, ngăn hiệu ứng trôi tích luỹ xuyên suốt nhiều task khác nhau trong cùng một fold — khác biệt bổ sung so với thiết kế gốc (vốn chỉ reset ở ranh giới lớn hơn).

---

## 7. Thuật toán test-time adaptation

*(Cập nhật v2.5: bổ sung vai trò teacher, bước cập nhật task-prompt có điều kiện; giữ nguyên cấu trúc 7 bước gốc, không thay đổi phần đã đúng)*

```
Input:  Test WSI X_i
        Merged model f_A(·, θ̃_{1:T}) với θ̃ gồm frozen weights + LN params {γ^l, β^l}
        Teacher model f_A(·, θ_teacher) — EMA của f_A, không nhận gradient trực tiếp
        Prompt embeddings {E^{C,t}} (frozen); E^T_source (frozen, neo);
        E^T (working copy, mutable qua EMA có điều kiện)
        Hyperparams: K%, E_0, α, β, γ_a, γ_m, δ_margin, β_anchor, η_teacher, η_prompt, n_steps

Output: Prediction ŷ_i

--- BƯỚC 1: FORWARD PASS (đa sub-bag) ---
1.1. Chia WSI X_i thành M sub-bag, mỗi sub-bag K_sub patches
1.2. Extract patch embeddings qua TITAN vision encoder (frozen)
1.3. Forward qua f_A (backbone/"student"): Z^(i) = f_A(sub-bag_i, θ̃_{1:T}) cho mỗi sub-bag
     [LN params trainable ở đây]

--- BƯỚC 2: DUAL-LEVEL PREDICTION ---
2.1. Task scores mỗi sub-bag: q^(i) = softmax(Z^(i) · E^{T⊤})
     t̂ = argmax(mean_i q^(i))  → predicted task (TCP), dùng embedding teacher để routing (ổn định hơn)
2.2. Class scores: p̃^(i) = softmax(Z^(i) · (E^{C,t̂})^⊤)

--- BƯỚC 3: CONFIDENCE FILTERING (intersection, v2.5) ---
3.1. Lọc theo class entropy: top-K% sub-bag entropy thấp nhất → tập A
3.2. Lọc theo task entropy: top-K% sub-bag entropy thấp nhất → tập B
3.3. B_K = A ∩ B  (v2.5: intersection thay vì union của v1.0)
3.4. Nếu B_K rỗng: fallback về sub-bag tin cậy nhất theo class

--- BƯỚC 4: TÍNH LOSS ---
4.1. Class-level: L_class = L_ent_class - L_div_class  (giữ nguyên v1.0, mục 6.1)
4.2. Task-level (v2.5, đã sửa):
     L_task = L_ent_task + γ_a · L_agree     (JSD giữa các sub-bag, THAY diversity)
4.3. Margin loss (mới, v2.5, tuỳ chọn bật/tắt):
     L_margin = ReLU(s_top2 - s_top1 + m)
4.4. Anti-forgetting: R_forget = Σ_l [(γ^l-γ^l_0)² + (β^l-β^l_0)²]  (L2-anchor đơn giản)
4.5. Total: L_TTA = L_class + α·L_task + γ_m·L_margin + β·R_forget

--- BƯỚC 5: UPDATE LN PARAMS (gradient descent) ---
5.1. Gradient: ∇_{γ^l,β^l} L_TTA
5.2. Update {γ^l, β^l} ← {γ^l, β^l} - lr · gradient  (n_steps bước)

--- BƯỚC 6: CẬP NHẬT TEACHER + TASK-PROMPT (EMA, không phải gradient — mới, v2.5) ---
6.1. Teacher EMA: θ_teacher ← η_teacher·θ_teacher + (1-η_teacher)·θ_student
6.2. Nếu margin(t̂) > δ_margin:
     E^T_t̂ ← (1-β_anchor)·[η_prompt·E^T_t̂ + (1-η_prompt)·Z̄] + β_anchor·E^T_t̂,source

--- BƯỚC 7: INFERENCE (dùng teacher, không dùng backbone/"student") ---
7.1. Re-forward qua teacher với LN params đã cập nhật (đồng bộ qua EMA bước 6.1)
7.2. Routing lại theo E^T hiện tại (có thể đã cập nhật ở bước 6.2)
7.3. ŷ_i = argmax p̃

--- BƯỚC 8: RESET POLICY ---
Option A (Episodic): Reset {γ^l, β^l}, teacher, và E^T về giá trị gốc sau mỗi slide
Option B (Continual): Giữ nguyên LN params và teacher đã adapt cho slide tiếp theo,
                      NHƯNG E^T được reset về E^T_source khi chuyển sang task mới
                      (v2.5: reset theo ranh giới task, không đợi đến hết fold)
→ Đã ablate cả hai; kết quả tốt nhất quan sát được ở chế độ continual cho LN/teacher
  kết hợp reset-theo-task cho E^T
```

---

## 8. Kế hoạch coding experiment

### Môi trường:
- Framework: PyTorch, dựa trên codebase gốc `caodoanh2001/MergeSlide`
- GPU: NVIDIA A100-SXM4 (80GB)
- Dataset: TCGA 6 tasks (BRCA, RCC, NSCLC, ESCA, TGCT, CESC) — cả IND và OOD setting (cross-site)

### Các bước implement (đã hoàn thành đến v2.5):

**Bước 1–7 (v1.0, đã hoàn thành):** LN param extraction, confidence filter, dual-level loss (bản gốc), L2-anchor regularizer, wrap thành module adapt, evaluate bACC/ACC/BWT/FGT.

**Bước 8 (v2.5, đã hoàn thành):** Sửa lỗi task-level diversity → agreement (JSD); chuyển confidence filtering từ union → intersection.

**Bước 9 (v2.5, đã hoàn thành):** Thêm teacher (EMA của backbone), dùng cho routing và inference cuối.

**Bước 10 (v2.5, đã hoàn thành):** Thêm cơ chế task-prompt embedding-space adaptation (EMA có gate + neo + reset theo task).

**Bước 11 (v2.5, đã hoàn thành):** Thêm task-level margin loss (tuỳ chọn, độc lập với agreement loss).

**Bước 12 (kết quả):** Evaluate trên IND, xác nhận cải thiện +1–2% bACC so với baseline ở TCP mode, khớp với kỳ vọng ở mục 9. **Chưa hoàn thành:** đánh giá đầy đủ trên IND reverse-order và OOD cross-site.

---

## 9. Baseline và ablation study

### Baselines cần so sánh:
| Baseline                                                      | Lý do                                                                      |
| --------------------------------------------------------------- | --------------------------------------------------------------------------|
| MergeSlide (no TTA)                                           | Upper bound reference                                                      |
| MergeSlide + TENT trên LN                                     | So sánh: chỉ class entropy, không có task entropy hay diversity/agreement            |
| MergeSlide + SHOT-IM trên LN                                  | So sánh: information maximization nhưng không có task-level loss           |
| MergeSlide + EATA trên LN                                     | So sánh: confidence filtering + anti-forgetting, nhưng không có dual-level |
| MergeSlide-TTA v1.0 (dual-level, task-diversity chưa sửa)     | **Baseline nội bộ quan trọng**: so sánh trực tiếp để định lượng đóng góp của việc sửa lỗi task-level ở v2.5 |
| MergeSlide-TTA (ours, không có $\mathcal{L}_{\text{task}}$)   | Ablation: chỉ class-level |
| MergeSlide-TTA (ours, không có $\mathcal{R}_{\text{forget}}$) | Ablation: không có regularizer |
| MergeSlide-TTA (ours, agreement nhưng không có teacher)       | **Mới, v2.5**: tách đóng góp của teacher EMA khỏi đóng góp của sửa loss task-level |
| MergeSlide-TTA (ours, teacher nhưng không có task-prompt adaptation) | **Mới, v2.5**: tách đóng góp của prompt-space adaptation khỏi teacher stabilization |
| MergeSlide-TTA (ours, task-prompt adaptation không có anchor, $\beta_{\text{anchor}}=0$) | **Mới, v2.5**: kiểm chứng vai trò của cơ chế neo — so với thiết kế SwapPrompt gốc không neo |
| MergeSlide-TTA Full (v2.5)                                    | Proposed method |

### Ablation study:
1. **Ablation thành phần loss:** Loại bỏ từng trong $\{\mathcal{L}_{\text{class}}, \mathcal{L}_{\text{task}}, \mathcal{L}_{\text{margin}}, \mathcal{R}_{\text{forget}}\}$.
2. **Ablation parameter group:** So sánh update LN vs. update full $f_A$ vs. update attention layers only.
3. **Ablation confidence threshold $E_0$:** Grid search $\{0.3, 0.5, 0.7, 0.9\} \times \log C$.
4. **Ablation reset policy:** Episodic vs. continual (cho LN/teacher), và có/không reset $\mathcal{E}^T$ theo ranh giới task.
5. **Ablation số bước update:** $n_{\text{steps}} \in \{1, 3, 5\}$ — **đã quan sát: sau khi sửa lỗi task-level (v2.5), bACC tăng đơn điệu theo $n_{\text{steps}}$ trong vùng nhỏ trước khi bão hoà, đúng như kỳ vọng lý thuyết — khác với hành vi giảm khi tăng $n_{\text{steps}}$ quan sát được ở thiết kế v1.0 (nguyên nhân đã xác định ở mục 6.2.1).**
6. **Ablation confidence filtering:** Union (v1.0) vs. intersection (v2.5).
7. **(Mới, v2.5) Ablation hệ số neo $\beta_{\text{anchor}} \in \{0, 0.3, 0.5, 0.7, 1.0\}$:** xác định điểm cân bằng giữa lợi ích thích nghi và rủi ro trôi.
8. **(Mới, v2.5) Ablation tính bền vững theo thứ tự (order-robustness):** so sánh kết quả khi giữ nguyên thứ tự slide trong luồng test tuần tự vs. khi thứ tự bị xáo trộn ngẫu nhiên trong cùng một task — kiểm định trực tiếp rủi ro echo-chamber nêu ở mục 6.4.

### Metrics đánh giá:
- **bACC** (Balanced Accuracy) — metric chính của MergeSlide paper
- **ACC** (Overall Accuracy)
- **BWT** (Backward Transfer) — đo forgetting sau TTA
- **$\Delta\text{bACC}_{\text{out-in}}$** — chênh lệch OOD vs. IND (metric domain robustness của MergeSlide)
- **Task prediction accuracy (routing accuracy)** — accuracy của TCP bước $\hat{t}$ trước khi predict class — **metric then chốt để tách bạch đóng góp của sửa lỗi task-level (mục 6.2.1) khỏi các thành phần khác**
- **Inference time** — overhead so với MergeSlide gốc (slides/s)
- **(Mới, v2.5) Độ dịch chuyển prompt $\|\mathcal{E}^T_t - \mathcal{E}^T_{t,\text{source}}\|_2$** cuối mỗi task — dùng để chẩn đoán mức độ "trôi" và tương quan với mức cải thiện, phục vụ kiểm chứng giả thuyết bổ sung ở mục 4

### Expected outcome (v1.0) — **Đã đạt được ở mức thận trọng của khoảng kỳ vọng:**
- MergeSlide-TTA Full > MergeSlide (no TTA) trên IND setting — **Confirmed: +1–2% bACC, TCP mode** (mục tiêu gốc đặt ra +1–3%, đã đạt cận dưới–giữa khoảng kỳ vọng trên IND; **chưa đánh giá OOD**).
- Task-level loss có đóng góp rõ ràng — **Confirmed, nhưng theo cơ chế khác với thiết kế gốc**: đóng góp chính đến từ việc sửa mục tiêu (diversity→agreement), không đơn thuần từ việc "có mặt" của task-level loss như giả định ban đầu.
- Regularizer $\mathcal{R}_{\text{forget}}$ quan trọng trong continual mode — **chưa tách bạch định lượng riêng ở v2.5**, cần ablation bổ sung (mục 9, ablation 1).

---

## 10. Rủi ro và cách kiểm chứng

| Rủi ro | Cách kiểm chứng | Trạng thái |
|---|---|---|
| LN params quá ít tham số, không đủ để bù domain shift | So sánh với "update attention layers" baseline; nếu gap lớn thì mở rộng scope | Chưa kiểm chứng đầy đủ; +1–2% đạt được có thể còn dư địa nếu mở rộng scope |
| Diversity loss không cân bằng được class imbalance của TCGA | Test riêng trên TGCT (rare task) và NSCLC (dominant task); dùng Tsallis entropy thay thế nếu cần | Áp dụng cho class-level (giữ nguyên); không áp dụng cho task-level nữa (đã thay bằng agreement) |
| Task prediction sai (TCP error) làm loss tính sai class | Log task prediction accuracy riêng; nếu > 10% error thì cần cơ chế fallback | **Confirmed đã từng xảy ra ở v1.0** (routing accuracy một task giảm còn ~30% do bug diversity); đã khắc phục ở v2.5, cần tiếp tục theo dõi trên OOD |
| Over-adaptation khi nhiều slides liên tiếp (continual mode) | Monitor bACC trên in-domain slides xen kẽ; nếu giảm, dùng episodic mode hoặc tăng $\beta$ | Chưa quan sát rõ ràng ở IND; cần theo dõi thêm ở OOD |
| Fisher importance estimate không chính xác trên WSI | So sánh với simple $\ell_2$ regularizer (không dùng Fisher); nếu tương đương thì dùng đơn giản hơn | **Đã quyết định: dùng L2-anchor đơn giản**, chưa định lượng "tương đương" bằng ablation trực tiếp với Fisher đầy đủ |
| Augmentation overhead (nếu dùng marginal entropy như TPT) | Tránh augmentation nhiều views; dùng patch-level/sub-bag confidence filtering thay thế | Đã áp dụng sub-bag filtering, không dùng augmentation kiểu TPT |
| **(Mới, v2.5) Rủi ro tự tham chiếu (echo-chamber) của task-prompt EMA** | Ablation $\beta_{\text{anchor}}=0$ vs. $>0$; test tính bền vững theo thứ tự slide (ablation 8, mục 9) | Đã có cơ chế giảm thiểu (anchor + reset theo task) trong thiết kế; **chưa có kết quả ablation định lượng riêng để xác nhận mức độ hiệu quả của anchor** |
| **(Mới, v2.5) Phụ thuộc thứ tự xử lý slide (order-dependence)** | So sánh kết quả với thứ tự gốc vs. thứ tự xáo trộn trong cùng một task | **Chưa kiểm chứng** — cần thực hiện trước khi đưa vào bản thảo nộp hội nghị, vì đây là câu hỏi phản biện điển hình cho các phương pháp online TTA có cập nhật tích luỹ theo luồng dữ liệu |
| **(Mới, v2.5) Độ trễ (lag) của teacher so với domain shift đột ngột** | So sánh bACC ở các slide đầu tiên của một task mới (ngay sau ranh giới domain shift) giữa có/không teacher | Chưa kiểm chứng |

---

## 11. Expected contributions

1. **Methodological:** Lần đầu tiên đề xuất test-time adaptation cho model merging trên WSI, với thiết kế dual-level entropy objective (class + task) phù hợp với TCP inference của MergeSlide. **Cập nhật v2.5:** bổ sung phát hiện quan trọng rằng mục tiêu diversity chuẩn của SHOT — vốn hiệu quả ở cấp class — **không** chuyển giao trực tiếp sang cấp task-routing khi các mẫu trong batch chia sẻ cùng nhãn thật (trường hợp multi-view/multi-sub-bag của cùng một instance); đây là một quan sát có thể tổng quát hoá sang các bài toán TTA khác có cấu trúc "nhiều view của cùng một thực thể" ngoài phạm vi WSI.

2. **Practical:** Chứng minh rằng chỉ cần cập nhật LN affine params (số lượng nhỏ, $\ll 1\%$ tổng params) kết hợp sửa đúng mục tiêu task-level là đủ để cải thiện đo lường được ($+1$–$2\%$ bACC) trên VLM-based WSI classification mà không cần source data. **Cập nhật v2.5:** bổ sung bằng chứng thực nghiệm cụ thể (không chỉ lý thuyết) cho tuyên bố này trên IND setting.

3. **Theoretical insight:** Bổ sung bằng chứng thực nghiệm rằng merging coefficients $\lambda_t$ nên được frozen trong TTA — không phải vì lý do computational, mà vì chúng encode cấu trúc orthogonal merging quan trọng cho stability. **Cập nhật v2.5:** phân biệt rõ giữa 3 mức độ "cứng" của các thành phần trong hệ thống — (i) không bao giờ được đổi ($\lambda_t$, class-prompt), (ii) có thể đổi tự do qua gradient (LN params), và (iii) có thể đổi có kiểm soát qua EMA-có-neo (task-prompt) — đây là một phân loại mới, chưa có trong v1.0, có thể là đóng góp lý thuyết độc lập về cách phân loại "độ cứng" của các thành phần trong một hệ thống model-merging khi thiết kế TTA.

4. **Benchmark:** Cung cấp IND/OOD TTA benchmark mới trên TCGA 6-task stream, có thể tái sử dụng cho các nghiên cứu sau, kèm theo baseline nội bộ v1.0-vs-v2.5 để minh hoạ cụ thể tác động của một lỗi thiết kế loss tưởng chừng nhỏ (áp dụng nhầm giả định diversity) lên kết quả cuối cùng — có giá trị tham khảo phương pháp luận cho các nghiên cứu TTA khác trên dữ liệu multi-instance.

---

## 12. Next steps

1. **Đã hoàn thành:** Reproduce baseline, implement dual-level loss v1.0, phát hiện và sửa lỗi task-level diversity→agreement, thêm teacher EMA, thêm task-prompt adaptation có neo, thêm margin loss, xác nhận cải thiện +1–2% trên IND (TCP mode).
2. **Tiếp theo — ưu tiên cao:** Chạy đầy đủ ablation tách bạch đóng góp riêng của từng thành phần mới (teacher / prompt-adaptation / margin loss / anchor) theo bảng mục 9, để biết chính xác thành phần nào đóng góp bao nhiêu vào +1–2% quan sát được.
3. **Tiếp theo — bắt buộc trước khi viết paper:** Kiểm định tính bền vững theo thứ tự xử lý slide (order-robustness, ablation 8 mục 9) — đây là câu hỏi phản biện gần như chắc chắn sẽ gặp phải với bất kỳ cơ chế online-EMA nào.
4. **Tiếp theo:** Đánh giá đầy đủ trên IND reverse-order và OOD cross-site — hiện mới chỉ có kết quả trên IND forward-order.
5. **Tiếp theo:** Định lượng lại vai trò của $\mathcal{R}_{\text{forget}}$ (L2-anchor đơn giản) so với phương án Fisher-weighted đầy đủ — quyết định giữ bản đơn giản hay đầu tư implement Fisher đầy đủ tuỳ vào kết quả ablation này.
6. **Tuỳ chọn, hướng mở rộng xa hơn:** Nếu ablation order-robustness cho thấy rủi ro thật, cân nhắc thay cơ chế EMA-tuần tự bằng cập nhật theo mini-epoch (gom một nhóm slide rồi mới cập nhật một lần, giảm phụ thuộc thứ tự từng slide đơn lẻ) — đây là hướng đi khác, chưa được thử nghiệm.

---

> **Ghi chú quan trọng (giữ nguyên từ v1.0):** Bài báo MergeSlide gốc đã có OOD evaluation với $\Delta\text{bACC}_{\text{out-in}} = -2.817\%$. Đây là baseline suy giảm cụ thể để TTA cần cải thiện. Mục tiêu thực tế: giảm gap này về $< 1.5\%$ mà không làm giảm IND performance.
>
> **Ghi chú bổ sung (v2.5):** Cải thiện +1–2% bACC hiện tại **mới được xác nhận trên IND**. Đây là điều kiện cần nhưng chưa đủ để kết luận về mục tiêu OOD nêu trên — domain shift trên OOD (cross-site) có thể có bản chất khác (ví dụ: shift về nhuộm màu/scanner) so với domain shift trong luồng IND (chủ yếu là task-switching giữa các cohort). Việc task-prompt adaptation có hiệu quả tương đương trên OOD hay không là câu hỏi thực nghiệm còn bỏ ngỏ, cần ưu tiên kiểm chứng trước khi khẳng định phương pháp giải quyết được gap OOD nêu trong ghi chú gốc.
