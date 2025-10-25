'use client';

import DetectorONNX from '../components/DetectorONNX';
import HeroSection from '../components/home/HeroSection';
import HowItWorksSection from '../components/home/HowItWorksSection';

export default function Page() {
  return (
    <main className="px-6 pb-24 pt-12 sm:px-8 lg:px-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-16">
        <HeroSection />
        <DetectorONNX />
        <HowItWorksSection />
      </div>
    </main>
  );
}
