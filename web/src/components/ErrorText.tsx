/** 操作失败（409、500、断网）时显示原因：没有它，按钮点了像什么都没发生。 */
export function ErrorText({ error }: { error: Error | null }) {
  return error ? <p className="mt-1 text-xs text-red-600">操作没有成功：{error.message}</p> : null
}
