# Design Template

## Problem

Xây dựng một research assistant nhận truy vấn dài, thu thập ngữ cảnh liên quan từ corpus cục bộ hoặc LLM provider, phân tích độ mạnh của bằng chứng, rồi tổng hợp câu trả lời cuối có nguồn tham chiếu.

## Why multi-agent?

Single-agent đủ cho câu hỏi ngắn, nhưng dễ trộn lẫn nhiều nhiệm vụ trong một prompt: tìm nguồn, phân tích và viết. Multi-agent giúp tách trách nhiệm rõ hơn, trace dễ hơn, và benchmark được từng bước để tìm failure mode chính xác.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn agent kế tiếp và quyết định khi nào dừng | `request`, `sources`, `research_notes`, `analysis_notes`, `final_answer`, `iteration` | route tiếp theo trong `route_history` | Router lặp vô hạn hoặc bỏ qua bước cần thiết |
| Researcher | Thu thập nguồn liên quan và tóm tắt bằng chứng | `request.query`, `request.max_sources` | `sources`, `research_notes` | Chọn nguồn yếu hoặc snippet quá ít thông tin |
| Analyst | Chuyển research notes thành insight có cấu trúc | `research_notes`, `sources`, `audience` | `analysis_notes` | Phân tích lặp lại nguồn mà không thêm giá trị |
| Writer | Tổng hợp câu trả lời cuối và đính kèm sources | `research_notes`, `analysis_notes`, `sources`, `audience` | `final_answer` | Trả lời dài dòng, thiếu citations hoặc lệch audience |

## Shared state

- `request`: giữ nguyên câu hỏi, audience và giới hạn số nguồn cho mọi agent.
- `iteration`: chặn workflow chạy vô hạn.
- `route_history`: debug được supervisor đã chọn gì qua từng vòng.
- `sources`: danh sách tài liệu mà Researcher tìm được để Analyst/Writer tái sử dụng.
- `research_notes`: bản tóm tắt evidence ban đầu.
- `analysis_notes`: insight đã được cấu trúc từ evidence.
- `final_answer`: output cuối cùng cho CLI và benchmark.
- `agent_results`: lưu nội dung và metadata như token/cost để đo benchmark.
- `trace`: ghi event định tuyến và thời lượng agent.
- `errors`: dành cho failure reporting và failure rate.

## Routing policy

Graph hiện tại:

`supervisor -> researcher -> analyst -> writer -> done`

Quy tắc:

- Nếu chưa có `sources` hoặc `research_notes` thì gọi `researcher`.
- Nếu đã có nghiên cứu nhưng chưa có `analysis_notes` thì gọi `analyst`.
- Nếu đã có phân tích nhưng chưa có `final_answer` thì gọi `writer`.
- Nếu đã có `final_answer` hoặc đạt `max_iterations` thì `done`.
- Sau khi hoàn tất, `critic` chạy như bước hậu kiểm nhẹ để ghi finding vào trace.

## Guardrails

- Max iterations: dùng `MAX_ITERATIONS`, mặc định `6`.
- Timeout: cấu hình `TIMEOUT_SECONDS`, hiện có ở settings để tích hợp provider thật.
- Retry: chưa retry network call riêng; khi tích hợp provider thật nên đặt retry trong `LLMClient` và `SearchClient`.
- Fallback: nếu không có OpenAI API key hoặc package `openai`, hệ thống dùng local deterministic fallback để vẫn chạy được.
- Validation: workflow ném `ValidationError` nếu kết thúc mà chưa có `final_answer`.

## Benchmark plan

- Query:
  - `Research GraphRAG state-of-the-art and write a 500-word summary`
  - `Compare single-agent and multi-agent workflows for customer support`
  - `Summarize production guardrails for LLM agents`
- Metric:
  - `latency_seconds`
  - `estimated_cost_usd`
  - `quality_score`
  - `citation_coverage`
  - `failure_rate`
- Expected outcome:
  - Baseline nhanh hơn hoặc tương đương nhưng chất lượng thấp hơn.
  - Multi-agent tốn nhiều bước hơn nhưng có trace rõ và giữ citations ổn định hơn.
