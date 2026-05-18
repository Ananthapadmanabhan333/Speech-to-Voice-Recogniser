import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Toaster } from 'react-hot-toast';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: {
    default: 'Neurolink - Adaptive Multimodal Communication System',
    template: '%s | Neurolink',
  },
  description:
    'An adaptive multimodal communication intelligence system that combines gesture recognition, speech analysis, and emotion detection for seamless human-computer interaction.',
  keywords: [
    'gesture recognition',
    'speech recognition',
    'emotion detection',
    'multimodal communication',
    'AI communication',
    'adaptive system',
  ],
  authors: [{ name: 'Neurolink Team' }],
  creator: 'Neurolink',
  publisher: 'Neurolink',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    siteName: 'Neurolink',
    title: 'Neurolink - Adaptive Multimodal Communication System',
    description:
      'An adaptive multimodal communication intelligence system for seamless human-computer interaction.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Neurolink - Adaptive Multimodal Communication System',
    description:
      'An adaptive multimodal communication intelligence system for seamless human-computer interaction.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.variable} antialiased`}>
        <Providers>{children}</Providers>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'hsl(0, 0%, 6%)',
              color: 'hsl(0, 0%, 95%)',
              border: '1px solid hsla(0, 0%, 100%, 0.06)',
              borderRadius: '12px',
              backdropFilter: 'blur(16px)',
            },
            success: {
              iconTheme: {
                primary: '#22c55e',
                secondary: '#052e16',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#450a0a',
              },
            },
          }}
        />
      </body>
    </html>
  );
}

function Providers({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
