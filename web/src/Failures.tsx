// The failure gallery. Images come from `python train/sample.py failures`
// (fixed seeds), and each caption describes exactly that image.
const base = import.meta.env.BASE_URL + "failures/";

const FAILURES: { images: string[]; title: string; why: string }[] = [
  {
    images: ["activities-1", "activities-2"],
    title: "“Activities” is not a thing",
    why: "Asked for Activities. The class mixes balls, medals, yarn and puzzle pieces, with the fewest examples (255). Even real Activities emoji are recognised only 12% of the time, so the model has no single idea to aim for.",
  },
  {
    images: ["symbols"],
    title: "It cannot spell",
    why: "Asked for Symbols. It learned “white glyph on a blue square” but not any actual character. At 32×32 with a few hundred examples, strokes become plausible-looking nonsense.",
  },
  {
    images: ["line-sketch-in", "line-sketch-out"],
    title: "Thin line drawings are ignored",
    why: "A line-art smiley, completed at strength 0.6. Training emoji are filled, coloured shapes; a thin black outline is mostly white pixels, so once noise is added almost nothing of it survives.",
  },
  {
    images: ["flag-in", "flag-out"],
    title: "Flags were never in the training set",
    why: "A three-stripe flag, completed as Travel & Places. Flags were left out of training on purpose, so the model bends the stripes into the nearest thing it knows, here something like a building.",
  },
  {
    images: ["five-steps"],
    title: "Five steps is too few",
    why: "Asked for Animals & Nature with 5 sampling steps. The colour and rough mass are right, but there are not enough steps to commit to edges. Recognisability is 82% at 5 steps against 87% at 10.",
  },
  {
    images: ["no-guidance"],
    title: "Guidance 0 ignores the class",
    why: "Asked for Food & Drink with guidance 0. The label is dropped entirely, so this is a sample from “any emoji”. Only 13% of such samples match the requested class, which is chance.",
  },
  {
    images: ["too-much-guidance"],
    title: "Guidance 8 burns the image",
    why: "Asked for Smileys with guidance 8. Pushing far past the conditional prediction over-commits: contrast is exaggerated and features turn harsh and dark.",
  },
  {
    images: ["objects"],
    title: "A confident blob",
    why: "Asked for Objects. One consistent colour, soft shading, a white background, and no identifiable object. The model has learned what emoji look like faster than what they are.",
  },
];

export default function Failures() {
  return (
    <section className="failures" id="failures">
      <h2>Where it fails</h2>
      <p>
        A 2-million-parameter model trained on 4,096 images has clear limits. These are picked to show them, not hidden to
        flatter it.
      </p>
      <div className="failure-grid">
        {FAILURES.map((f) => (
          <figure key={f.title}>
            <div className="failure-images">
              {f.images.map((name, i) => (
                <img key={name} src={`${base}${name}.png`} width={256} height={256} loading="lazy"
                  alt={f.images.length === 2 && f.images[0].endsWith("-in") ? (i === 0 ? "Input sketch" : "Model output") : `Generated sample: ${f.title}`} />
              ))}
            </div>
            <figcaption><strong>{f.title}.</strong> {f.why}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
