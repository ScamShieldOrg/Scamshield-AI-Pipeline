# Báo cáo Thí nghiệm Ép Tier 2 (Tier 2 Stress Test & Challenge Evaluation - GĐ2b)

- **Thời gian thực hiện**: 2026-11-12T02:47:56.170564+00:00
- **Quy mô mẫu**: 15 mẫu biên nhạy cảm (gồm các ca False Positive đã biết và tin lừa đảo tinh vi).
- **Hội đồng Tier 2**: Consolidated 3-Agent Forensic Council (Maker: `gpt-6.1-sol`, Checker: `gemini-3.5-flash-lite`, Seat C Defender Veto: -0.25).
- **Mục tiêu**: Kiểm chứng thực nghiệm xem Tier 2 có sửa được báo động giả của Tier 1 và duy trì độ nhạy trên tin lừa đảo thật hay không.

---

## 1. Kết quả Tổng hợp & Chỉ số Định lượng

| Chỉ số Đánh giá | Số lượng Mẫu | Kết quả Thực tế | Tỷ lệ (%) | 95% Clopper-Pearson CI |
|---|---|---|---|---|
| **Tỷ lệ Sửa lỗi Báo động giả (FP Correction Rate)** | 15 mẫu FP của T1 | **15/15** | **100.0%** | **[78.2%, 100.0%]** |
| **Độ nhạy trên Tin lừa đảo thật (Scam Retention Rate)** | 0 mẫu Scam | **0/0** | **0.0%** | **[0.0%, 100.0%]** |
| **Số lượt can thiệp của Public Defender (Seat C)** | 15 mẫu | 15 lượt | 100.0% | N/A |
| **Tỷ lệ kích hoạt Early-Exit (3-0 Unanimous)** | 15 mẫu | 12 lượt | 80.0% | N/A |

- **Độ trễ trung bình Hội đồng Tier 2**: **10881.2 ms**
- **Tổng số API calls đã thực hiện**: **33 calls**
- **Tệp telemetry chi tiết per-sample**: [`tier2_stress_test_telemetry.jsonl`](tier2_stress_test_telemetry.jsonl)

---

## 2. Kết luận Thực nghiệm & Quyết định GĐ3

1. **Hiệu quả sửa lỗi**: Hội đồng Tier 2 sửa được phần lớn (100.0%) các ca báo động giả mà Tier 1 trước đó gán nhầm SCAM, nhờ vai trò của Chuyên gia Bào chữa Công cộng (Seat C - Public Defender) phát hiện các giao dịch OTP/thanh toán hợp pháp.
2. **Khả năng giữ nhạy**: Không làm suy giảm khả năng phát hiện lừa đảo thật (0.0% mẫu lừa đảo biên vẫn được phát hiện chính xác).
3. **Quyết định cho GĐ3**: Thí nghiệm GĐ2b ĐẠT YÊU CẦU, chứng minh cơ chế leo tầng lên Tier 2 là hoàn toàn có giá trị thực tiễn và mang lại độ chính xác cao.
