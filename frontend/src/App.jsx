import { useRef, useState } from "react";
import {
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  FileImage,
  Loader2,
  RotateCcw,
  Telescope,
  Upload,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const selectFile = (selected) => {
    if (!selected) return;

    const allowed = ["image/jpeg", "image/png", "image/webp"];

    if (!allowed.includes(selected.type)) {
      setError("Please select a JPG, PNG, or WEBP image.");
      return;
    }

    if (selected.size > 15 * 1024 * 1024) {
      setError("Image exceeds the 15 MB limit.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setError(
        "Analysis could not be completed. Make sure the AstroLens API is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-[#05070a] text-white">
      <div className="fixed inset-0 pointer-events-none bg-grid opacity-[0.025]" />

      <header className="relative z-10 border-b border-white/[0.07] bg-[#05070a]/90 backdrop-blur-md">
        <div className="mx-auto flex h-18 max-w-7xl items-center justify-between px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center border border-blue-500/30 bg-blue-500/10">
              <Telescope className="h-5 w-5 text-blue-400" />
            </div>

            <div>
              <div className="text-sm font-semibold tracking-[0.16em]">
                ASTROLENS
              </div>
              <div className="text-[9px] tracking-[0.2em] text-slate-500">
                ASTRONOMICAL ANALYSIS SYSTEM
              </div>
            </div>
          </div>

          <div className="hidden items-center gap-8 text-[11px] tracking-[0.14em] text-slate-500 sm:flex">
            <span className="text-slate-300">OBSERVATORY</span>
            <span>ABOUT</span>
            <span className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400 shadow-[0_0_8px_#3b82f6]" />
              SYSTEM ONLINE
            </span>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-6 pb-20 lg:px-8">
        {!result ? (
          <UploadView
            file={file}
            preview={preview}
            loading={loading}
            dragging={dragging}
            error={error}
            inputRef={inputRef}
            onSelect={selectFile}
            onAnalyze={analyze}
            onDragging={setDragging}
          />
        ) : (
          <ResultsView
            result={result}
            preview={preview}
            reset={reset}
          />
        )}
      </main>

      <footer className="relative z-10 border-t border-white/[0.06]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 text-[10px] tracking-[0.12em] text-slate-600 lg:px-8">
          <span>ASTROLENS AI · IMAGE ANALYSIS</span>
          <span>LOCAL INFERENCE PIPELINE</span>
        </div>
      </footer>
    </div>
  );
}

function UploadView({
  file,
  preview,
  loading,
  dragging,
  error,
  inputRef,
  onSelect,
  onAnalyze,
  onDragging,
}) {
  return (
    <section className="pt-16 sm:pt-24">
      <div className="mb-12 max-w-3xl">
        <div className="mb-5 flex items-center gap-3 text-[11px] font-medium tracking-[0.22em] text-blue-400">
          <span className="h-px w-8 bg-blue-500" />
          ASTRONOMICAL IMAGE ANALYSIS
        </div>

        <h1 className="max-w-3xl text-4xl font-semibold leading-[1.08] tracking-[-0.03em] text-white sm:text-6xl">
          See deeper into
          <br />
          <span className="text-slate-500">the night sky.</span>
        </h1>

        <p className="mt-6 max-w-xl text-sm leading-7 text-slate-400 sm:text-base">
          Upload an astronomical image to identify the object, generate an
          image caption, and retrieve relevant astronomical observations.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          onDragging(true);
        }}
        onDragLeave={() => onDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          onDragging(false);
          onSelect(e.dataTransfer.files?.[0]);
        }}
        onClick={() => !file && inputRef.current?.click()}
        className={[
          "relative overflow-hidden border transition-all duration-300",
          dragging
            ? "border-blue-400 bg-blue-500/[0.06]"
            : "border-white/[0.1] bg-[#080c11]",
          !file ? "cursor-pointer hover:border-blue-500/40" : "",
        ].join(" ")}
      >
        <div className="absolute left-0 top-0 h-px w-32 bg-blue-500" />
        <div className="absolute right-0 top-0 h-px w-16 bg-white/20" />

        {preview ? (
          <div className="grid lg:grid-cols-[1fr_360px]">
            <div className="relative flex min-h-[420px] items-center justify-center bg-black p-6">
              <img
                src={preview}
                alt="Selected astronomical image"
                className="max-h-[560px] max-w-full object-contain"
              />

              <div className="absolute left-5 top-5 border border-white/10 bg-black/70 px-3 py-2 text-[9px] tracking-[0.16em] text-slate-400 backdrop-blur">
                PREVIEW
              </div>
            </div>

            <div
              className="flex flex-col justify-between border-t border-white/[0.08] p-7 lg:border-l lg:border-t-0"
              onClick={(e) => e.stopPropagation()}
            >
              <div>
                <div className="flex items-center gap-2 text-[10px] tracking-[0.18em] text-blue-400">
                  <FileImage className="h-4 w-4" />
                  SELECTED IMAGE
                </div>

                <h2 className="mt-5 break-words text-lg font-medium text-white">
                  {file?.name}
                </h2>

                <p className="mt-2 text-xs text-slate-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>

              <div className="mt-10 space-y-3">
                <button
                  onClick={onAnalyze}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-3 bg-blue-600 px-5 py-3.5 text-xs font-semibold tracking-[0.12em] transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      ANALYZING
                    </>
                  ) : (
                    <>
                      ANALYZE IMAGE
                      <ChevronRight className="h-4 w-4" />
                    </>
                  )}
                </button>

                <button
                  onClick={() => inputRef.current?.click()}
                  className="w-full border border-white/10 px-5 py-3.5 text-xs tracking-[0.12em] text-slate-400 transition hover:border-white/20 hover:text-white"
                >
                  CHOOSE ANOTHER
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex min-h-[390px] flex-col items-center justify-center px-6 py-16 text-center">
            <div className="mb-7 flex h-16 w-16 items-center justify-center border border-blue-500/20 bg-blue-500/[0.05]">
              <Upload className="h-6 w-6 text-blue-400" />
            </div>

            <h2 className="text-xl font-medium">
              Upload an astronomical image
            </h2>

            <p className="mt-3 max-w-md text-sm leading-6 text-slate-500">
              Drag and drop an image here, or click to browse your observation
              files.
            </p>

            <div className="mt-7 flex items-center gap-3 text-[9px] tracking-[0.18em] text-slate-600">
              <span>JPG</span>
              <span>·</span>
              <span>PNG</span>
              <span>·</span>
              <span>WEBP</span>
              <span>·</span>
              <span>MAX 15 MB</span>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => onSelect(e.target.files?.[0])}
            />
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 border border-red-500/20 bg-red-500/[0.05] px-5 py-4 text-xs text-red-300">
          {error}
        </div>
      )}

      <div className="mt-8 grid gap-px border border-white/[0.07] bg-white/[0.07] sm:grid-cols-3">
        <Feature
          number="01"
          title="IDENTIFY"
          description="Zero-shot astronomical object recognition."
        />
        <Feature
          number="02"
          title="DESCRIBE"
          description="AI-generated visual caption from the image."
        />
        <Feature
          number="03"
          title="EXPLORE"
          description="Relevant astronomy knowledge and observations."
        />
      </div>
    </section>
  );
}

function Feature({ number, title, description }) {
  return (
    <div className="bg-[#080c11] p-6">
      <div className="text-[10px] tracking-[0.16em] text-blue-500">
        {number}
      </div>

      <div className="mt-5 text-xs font-semibold tracking-[0.15em] text-slate-300">
        {title}
      </div>

      <p className="mt-2 text-xs leading-5 text-slate-600">
        {description}
      </p>
    </div>
  );
}

function ResultsView({ result, preview, reset }) {
  const top = result.recognition?.[0];
  const confidence = top ? top.confidence * 100 : 0;

  return (
    <section className="pt-12 sm:pt-16">
      <div className="mb-10 flex flex-col justify-between gap-5 border-b border-white/[0.08] pb-7 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-3 text-[10px] tracking-[0.2em] text-blue-400">
            <CheckCircle2 className="h-4 w-4" />
            ANALYSIS COMPLETE
          </div>

          <h1 className="mt-4 text-3xl font-semibold tracking-[-0.02em] sm:text-4xl">
            {top?.label || "Unknown Object"}
          </h1>

          <p className="mt-2 text-xs tracking-[0.12em] text-slate-600">
            ASTROLENS OBSERVATION REPORT
          </p>
        </div>

        <button
          onClick={reset}
          className="flex items-center justify-center gap-2 border border-white/10 px-4 py-2.5 text-[10px] tracking-[0.14em] text-slate-400 transition hover:border-white/20 hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          NEW ANALYSIS
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <div className="relative overflow-hidden border border-white/[0.08] bg-black">
          {preview && (
            <img
              src={preview}
              alt="Analyzed astronomical object"
              className="max-h-[620px] w-full object-contain"
            />
          )}

          <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between border-t border-white/10 bg-black/80 px-5 py-3 backdrop-blur">
            <span className="text-[9px] tracking-[0.16em] text-slate-500">
              SOURCE IMAGE
            </span>
            <span className="text-[9px] tracking-[0.12em] text-slate-600">
              LOCAL ANALYSIS
            </span>
          </div>
        </div>

        <div className="border border-white/[0.08] bg-[#080c11]">
          <div className="border-b border-white/[0.08] px-6 py-5">
            <div className="text-[9px] tracking-[0.2em] text-slate-600">
              OBJECT IDENTIFICATION
            </div>
          </div>

          <div className="p-6">
            <div className="text-2xl font-semibold leading-tight text-white">
              {top?.label || "Unknown"}
            </div>

            <div className="mt-2 text-xs text-slate-600">
              Primary CLIP recognition
            </div>

            <div className="mt-8">
              <div className="flex items-end justify-between">
                <span className="text-[9px] tracking-[0.16em] text-slate-600">
                  CONFIDENCE
                </span>

                <span className="text-2xl font-medium text-blue-400">
                  {confidence.toFixed(2)}%
                </span>
              </div>

              <div className="mt-3 h-1 bg-white/[0.07]">
                <div
                  className="h-full bg-blue-500 transition-all duration-700"
                  style={{ width: `${Math.min(confidence, 100)}%` }}
                />
              </div>
            </div>

            <div className="mt-10 border-t border-white/[0.07] pt-5">
              <div className="text-[9px] tracking-[0.16em] text-slate-600">
                ALTERNATIVE IDENTIFICATIONS
              </div>

              <div className="mt-4 space-y-3">
                {result.recognition?.slice(1, 5).map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className="text-slate-400">{item.label}</span>
                    <span className="font-mono text-slate-600">
                      {(item.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <section className="mt-8 grid gap-6 lg:grid-cols-2">
        <ReportBlock title="AI CAPTION">
          <p className="text-sm leading-7 text-slate-400">
            {result.caption || "No caption available."}
          </p>
        </ReportBlock>

        <ReportBlock title="SCIENTIFIC OBSERVATION">
          <p className="whitespace-pre-line text-sm leading-7 text-slate-400">
            {result.scientific_observation ||
              result.observation ||
              "No observation available."}
          </p>
        </ReportBlock>
      </section>

      {result.facts?.length > 0 && (
        <section className="mt-10">
          <SectionHeading
            eyebrow="OBJECT NOTES"
            title="Interesting facts"
          />

          <div className="mt-5 grid gap-px border border-white/[0.07] bg-white/[0.07] md:grid-cols-3">
            {result.facts.map((fact, index) => (
              <div key={index} className="bg-[#080c11] p-6">
                <div className="font-mono text-[10px] text-blue-500">
                  0{index + 1}
                </div>

                <p className="mt-5 text-sm leading-6 text-slate-400">
                  {fact}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {result.knowledge?.length > 0 && (
        <section className="mt-12">
          <SectionHeading
            eyebrow="ARCHIVE"
            title="Related astronomical observations"
          />

          <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {result.knowledge.map((item, index) => (
              <article
                key={index}
                className="group border border-white/[0.08] bg-[#080c11] p-6 transition hover:border-blue-500/30"
              >
                <div className="flex items-start justify-between">
                  <span className="text-[9px] tracking-[0.15em] text-blue-500">
                    NASA / APOD
                  </span>

                  <ArrowUpRight className="h-4 w-4 text-slate-700 transition group-hover:text-blue-400" />
                </div>

                <h3 className="mt-5 text-base font-medium leading-6 text-slate-200">
                  {item.title || "Astronomical Observation"}
                </h3>

                <p className="mt-3 text-xs leading-6 text-slate-500">
                  {item.summary}
                </p>

                <div className="mt-6 flex items-center justify-between border-t border-white/[0.06] pt-4">
                  <span className="font-mono text-[9px] text-slate-700">
                    {item.date || "ARCHIVE"}
                  </span>

                  {item.source_url && (
                    <a
                      href={item.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[9px] tracking-[0.12em] text-blue-500 hover:text-blue-400"
                    >
                      OPEN SOURCE
                    </a>
                  )}
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      <div className="mt-12 border-t border-white/[0.07] pt-8">
        <button
          onClick={reset}
          className="flex items-center gap-2 text-xs tracking-[0.14em] text-slate-500 transition hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          ANALYZE ANOTHER IMAGE
        </button>
      </div>
    </section>
  );
}

function ReportBlock({ title, children }) {
  return (
    <div className="border border-white/[0.08] bg-[#080c11]">
      <div className="border-b border-white/[0.08] px-6 py-4">
        <span className="text-[9px] tracking-[0.2em] text-slate-600">
          {title}
        </span>
      </div>

      <div className="p-6">{children}</div>
    </div>
  );
}

function SectionHeading({ eyebrow, title }) {
  return (
    <div>
      <div className="flex items-center gap-3 text-[9px] tracking-[0.2em] text-blue-500">
        <span className="h-px w-6 bg-blue-500" />
        {eyebrow}
      </div>

      <h2 className="mt-3 text-2xl font-medium tracking-tight text-slate-200">
        {title}
      </h2>
    </div>
  );
}