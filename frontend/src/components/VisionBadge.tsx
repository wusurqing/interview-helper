import { Eye } from 'lucide-react'

export default function VisionBadge() {
  return (
    <span className="inline-flex shrink-0 items-center gap-0.5 rounded-md bg-accent-blue/10 px-1.5 py-0.5 text-[10px] font-medium leading-4 text-accent-blue">
      <Eye className="h-3 w-3" aria-hidden /> 识图
    </span>
  )
}
