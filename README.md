# Allan's ComfyUI nodes

These are not the nodes you're looking for.

This is actually a private set of nodes (though obviously not confidential) that need to be hosted in a repo that does not require authentication.

I have no plans to distribute and/or advertise this project.

Also, I see people starring and cloning this repo. I'm grateful, but the reason these nodes aren't officially "released" or registered over at the ComfyUI registry is because I want the freedom to break things. Sometimes renaming nodes, other times outright deleting them.

Just so you know...

## Nodes

This list is more for my benefit. If anything looks interesting, let me know and I may break it out into its own (smaller) package.

### Parameter Experimentation

* `MakeImageGrid` Takes a list or batch of images that may be of varying dimensions, resizes them, and arranges them into a grid
* `WriteTextImage` Takes an image and a value (which can be anything that can be represented as a string) and writes it either to the image's alpha channel or to the image itself
* `FloatLatch` & `IntegerLatch` Part of a my series of "latch" nodes which capture an optional input and pass it through to the output. The captured value can be frozen and edited. Note that the `TextLatch` is over at my [ComfyUI-YALLM-node](https://github.com/asaddi/ComfyUI-YALLM-node) suite.
* `ImageBuffer` Temporary storage for images. Has two modes: "accumulate" and "release." While accumulating, it will save all incoming images. Finally, when you switch to "release," it will output all images. I just got tired of saving test images across multiple prompts and then using the batch loader to do something with them all (e.g. making a grid).

### Lists

Nodes that generate lists. Useful for workflows that generate a series of images that vary in something or another (a strength value, seed, etc.). Queue once, generate many, basically. (Also useful for XY plotting...)

* `ImageSequenceList` Takes one or more image inputs, combines them into a sequence (e.g. `[a, b, c]`) and then optionally repeats the sequence. I've found this useful when multiple images are generated at once (e.g. multiple samplers in parallel). It allows me to explicitly specify the order that they will appear in the sequence which helps when generating grids.
* `StringSequenceList` Takes one or more string inputs, combines them into a sequence (e.g. `[a, b, c]`) and then optionally repeats the sequence. Seems useful for testing prompt variants in a tightly-controlled manner (i.e. no need to step through wildcard combos).
* `RepeatStringList` / `RepeatIntList` / `RepeatFloatList` Takes a list of values (e.g. `[a, b, c]`) and repeats them in specific ways. For example, repeating 3 times consecutively: `[a, a, a, b, b, b, c, c, c]` or sequentially: `[a, b, c, a, b, c, a, b, c]`
* `FloatList` / `FloatListStepSize` Takes a start and end value and outputs a list of floats that iterates through those values. Great for parameter experimentation (e.g. LoRA strength, control net strength, Flux Redux strength).
* `SeedList` Generates a list of seeds (random numbers). The `SeedList` node itself takes a seed so it is reproducible & deterministic.

### Image Utilities

* `BatchImageLoader` My version of a batch image loader. Batches images of like dimensions together, generates lists, supports directory recursion. Supports alpha channels/masks as well as JSON extraction (i.e. prompt/workflow data)
   * `JSONExtractString` & `JSONExtractNumber` Uses [JMESPath](https://jmespath.org/) to extract values from JSON (currently, only from the `BatchImageLoader`)
   * `PathJoin`, `PathSplit` & `PathRelativeTo`. A set of string utilities meant for manipulating & extracting information from file paths. Usually used with `BatchImageLoader`.
* `MaskBlur` Basically just a glorified group node to apply Gaussian blur to a mask
* `ImageDimensions` Extracts width/height from an image (also displays it as a node badge, so you can see the dimensions at a glance without any other nodes)
* `FlattenImageAlpha` Essentially composites an image with alpha onto a solid background. Can be easily done with Core nodes + `ImageDimensions`, but this seemed like a convenient shorthand.
* `ImageCropSquare` Crops the input image into a square. Aside from center cropping, supports a crop focusing on one of the sides. Under the hood, it simply expands into the Core Image Crop node. Useful for CLIP Vision/IPAdapter usage as it avoids surprises from using a non-square source.
* `SaveComicBookArchive` Saves images into a ComicBookArchive (it's just a zip file).

### Utilities

* `PrivateLoraStack` My super simple version of a LoRA stack. Uses ~~domain~~ node expansion and chains the Core LoRA loader appropriately.
* `ResolutionChooser` Selects resolutions based on: desired aspect ratio, desired orientation, mebipixel budget (mebi = 2^20 = 1,048,576, not exactly 1 million), divisor/multiple which is typically 64 for most diffusion models (i.e. resolution rounded to 64 pixels)
* `EmptyLatentImageSelector` Another glorified group node that picks the correct Core EmptyLatent for the selected model
* `PrivateSeed` My own seed generator with history
* `MixNoise` My own take on noise mixing. The mask is much more significant and rather than doing a "blend" (`noise1 * (1.0 - weight) + noise2 * weight`), instead `noise1` is always at full strength and `noise2` is used to perturb it. Currently always normalizes the mixed output.
* `RandomCombo2` Probabalistically chooses one of two combo options (both customizable, along with their probabilities). Has a seed input, of course. I use it to randomly switch image orientation between portrait & landscape.

### Buses (Combiners/Splitters)

I kind of loathe the idea of "buses" as it obfuscates all the glorious noodles and makes it hard to see what connects to what. But since I like building modules (workflow fragments) and then quickly gluing them together, it seems kind of a necessity.

All bus nodes are fully "lazy" -- if nothing uses the output downstream, then the input won't be evaluated.

* `SimpleBus` Basic bus for model/VAE/latent/guider. Yes, that last one is an odd choice, but I like using the advanced custom sampler and it makes the most sense for the modules I've built.
* `ControlBus` ControlNet-oriented bus for positive condition/negative condition/image/mask. I also use it to link the prompting modules with the main model modules.

### Prompting

* `VersatileTextEncode` A glorified group node that uses Core nodes underneath (via node expansion). Meant for use with standard (SD1/SDXL/Flux), Pony, SD3.5 prompting and reconfigures appropriately. Did you know that to use "BREAK" in the Pony preamble ("score_9, score_8_up, etc."), you have to use a `ConditioningConcat` node? Yeah, this takes care of that.
* `PresetText` Basically a text input node that allows you to load from presets (stored as a YAML file). Useful for positive/negative prompts, or even just notes.
* `ReproducibleWildcards` My take on seeded wildcards. It's not really random, the wildcard choices it makes are based on the seed. So your seed better be random! This means if you increment the seed by 1 you can also step through through all choices combinatorically. (~~I may know a little bit about combinatorics...~~)

### Switches

* `ImageRouter` Routes an incoming image to one of two outputs based on tags (from the WDv3 Tagger below). Can perform argmax (i.e. select the tag with the highest probability) or simple threshold testing. Basic RegExp knowledge required.
* `ModelSwitch` / `VAESwitch` / `ImageSwitch` / `ImageMaskSwitch` Switches that simply select the first non-null input. `ImageMaskSwitch`, of course, will select the first non-null image and its associated mask

### Tabular Data

An experimental set of nodes for creating tabular data, specifically for generating CSV or TSV files.

* `TabularJoin` Combines multiple inputs into a single row of data. Can be used recursively for an arbitrary number of columns.
* `TabularSave` Saves tabular data into a CSV or TSV file.

### Debugging

* `DumpToConsole` Pretty prints the input to the console
* `ListCounter` Outputs the number of elements in the input. Useful for debugging list-based workflows.

### Third-Party Models

I'm pretty sure nodes for these exist elsewhere, but I didn't want to bring in node suites with 50+ nodes and dozens of dependencies just to use them.

See below for links to the models & the original (Python) projects.

* Depth Anything V2 support
* WDv3 Tagger support
* Perceptual Similarity (LPIPS) support

## Notes

* The `depth_anything_v2` package was copied verbatim from https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
* The `Write Text to Image` node uses the Calibri Bold font (`calibrib.ttf`, which I chose not to include). It may easily be acquired from any Windows installation under `C:\WINDOWS\FONTS`

## Projects & Models Used

* https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3
   * https://github.com/neggles/wdv3-timm
* https://huggingface.co/depth-anything/Depth-Anything-V2-Large
   * https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
* https://richzhang.github.io/PerceptualSimilarity
   * https://github.com/richzhang/PerceptualSimilarity
