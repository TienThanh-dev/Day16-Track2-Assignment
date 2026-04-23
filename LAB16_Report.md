# BÁO CÁO LAB 16 — Cloud AI Environment Setup (CPU + LightGBM)

> **Phương án thực hiện:** CPU Instance + LightGBM (không dùng GPU vì tài khoản AWS Free Tier không được cấp quota GPU)

---

## 1. Thông tin hạ tầng

| Thông số | Giá trị |
|---|---|
| Instance type | m7i-flex.large |
| AMI | Amazon Linux 2023 (auto-lookup bằng Terraform data source) |
| Bastion public IP | `100.29.184.108` |
| CPU Node private IP | `10.0.10.224` |
| ALB DNS | `ai-inference-alb-f6682353-662341349.us-east-1.elb.amazonaws.com` |
| VPC CIDR | `10.0.0.0/16` |
| Region | `us-east-1` |

---

## 2. Kết quả Benchmark — LightGBM Credit Card Fraud Detection

| Thông số | Giá trị |
|---|---|
| Dataset | Credit Card Fraud (Kaggle — 284,807 giao dịch) |
| Features | 28 (V1–V28) |
| **Data load time** | **0.521s** |
| **Training time** | **8.847s** |
| Best iteration | 0 (early stopping) |
| **AUC-ROC** | **0.8958** |
| **Accuracy** | **93.44%** |
| **F1-Score** | **0.0430** |
| **Precision** | **2.21%** |
| **Recall** | **85.71%** |
| Inference latency (1 row) | 0.266ms |
| Inference throughput (1000 rows) | 0.004s |

**File kết quả:** `/home/ec2-user/ml-benchmark/benchmark_result.json` (trên CPU Node)

---

## 3. Cold Start Time

> **Ghi lại số phút thực tế: ~9 phút.**

---

## 4. Tại sao dùng CPU thay vì GPU?

Tài khoản AWS Free Tier mặc định không được cấp quota GPU (hạn mức `g4dn.xlarge` = 0 vCPU). Yêu cầu tăng quota GPU bị trì hoãn hoặc từ chối. Phương án CPU + LightGBM đảm bảo đầy đủ quy trình: **IaC → Cloud Instance → Training → Inference → Billing check**.

---

## 5. Phân tích kết quả

- **AUC-ROC 0.8958**: Mô hình phân biệt tốt giữa giao dịch fraud và normal.
- **Precision thấp (2.21%)**: Đặc trưng của bài toán highly imbalanced — chỉ 0.173% transactions là fraud (492/284,807). Mô hình ưu tiên recall cao (85.71%) để bắt hầu hết gian lận.
- **Inference 0.266ms/row**: Phù hợp cho real-time fraud detection.
- **Training 8.847s / Data load 0.521s**: Nhanh trên CPU nhờ LightGBM tối ưu.

---

## 6. Chi phí thực tế sau 1 giờ

Tài khoản AWS Free Tier nên **chi phí phát sinh bằng $0** trong giới hạn Free Tier.

| Dịch vụ | Loại | Free Tier limit | Chi phí thực tế |
|---|---|---|---|
| EC2 — CPU Node | `m7i-flex.large` | 750h/tháng | ~$0 |
| EC2 — Bastion | `t3.micro` | 750h/tháng | ~$0 |
| NAT Gateway | per GB | 100 GB/tháng | ~$0 |
| ALB | Application LB | 750h/tháng | ~$0 |
| **Tổng** | | | **~$0 (trong giới hạn Free Tier)** |
Vì dùng free nên không thấy data bill nhảy tiền 

---

## 7. Báo cáo ngắn (5–10 dòng)

Phương án CPU + LightGBM được sử dụng thay vì GPU vì tài khoản AWS Free Tier không được cấp quota GPU và chưa accept thanh toán card nên vào chế độ only Free Tier. Trên instance `m7i-flex.large` (8 GB RAM), LightGBM train 284,807 giao dịch credit card trong **8.847s**, đạt AUC-ROC **0.8958** và Recall **85.71%** — bắt được phần lớn giao dịch gian lận. Inference nhanh (**0.266ms/row**) phù hợp cho real-time fraud detection. Chi phí nằm trong giới hạn **Free Tier (~$0)**. Toàn bộ pipeline: Terraform IaC → Cloud Instance → ML Training → API Inference → Billing check đã hoàn thành đầy đủ.

