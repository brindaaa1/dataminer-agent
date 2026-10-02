import type { Snapshot } from "@/api"

export const baseSnap: Snapshot = {
  id: "t1", title: "hotel · 10-02", status: "intake", stage: "intake", example: false, readonly: false, busy: false,
  error: null, provider: "mock", langfuse_url: null, intake_langfuse_url: null, note: null,
  data: { csv_name: "hotel_bookings_sample.csv", description: "预测是否取消", n_rows: 7997, n_cols: 32 },
  intake: null, run: null, artifacts: [], last_event_id: 0,
}
