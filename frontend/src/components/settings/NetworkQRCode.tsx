import { useState, useEffect } from 'react'
import { Smartphone } from 'lucide-react'
import QRCode from 'qrcode'
import { Section } from './shared'
import { buildApiUrl } from '@/lib/backendUrl'

export default function NetworkQRCode() {
  const [qrSrc, setQrSrc] = useState<string | null>(null)
  const [networkUrl, setNetworkUrl] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadNetworkQr() {
      try {
        const response = await fetch(buildApiUrl('/api/network-info'))
        if (cancelled) return
        const data = await response.json() as { url?: unknown }
        if (cancelled) return
        const url = typeof data.url === 'string' ? data.url : ''
        if (!url) return

        setNetworkUrl(url)
        const src = await QRCode.toDataURL(url, {
          width: 200,
          margin: 2,
          color: { dark: '#000000', light: '#ffffff' },
        })
        if (cancelled) return
        setQrSrc(src)
      } catch {
        // Network QR is optional; keep settings usable if LAN discovery fails.
      }
    }

    void loadNetworkQr()
    return () => {
      cancelled = true
    }
  }, [])

  if (!networkUrl) return null

  return (
    <Section title="手机扫码访问" icon={<Smartphone className="w-3.5 h-3.5" />} keywords="手机 扫码 二维码 QR code network wifi 局域网">
      <div className="flex flex-col items-center gap-2">
        {qrSrc && <img src={qrSrc} alt="QR Code" className="rounded-lg" width={160} height={160} />}
        <p className="text-[11px] text-accent-blue break-all text-center select-all">{networkUrl}</p>
        <p className="text-[10px] text-text-muted text-center">手机和电脑需在同一 WiFi 下，音频在电脑端采集</p>
      </div>
    </Section>
  )
}
