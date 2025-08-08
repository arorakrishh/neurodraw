# NeuroDraw - Neural Style Transfer Implementation

An implementation of [neural style][paper] in TensorFlow.

This implementation is designed to be simple and efficient, leveraging TensorFlow's API and [automatic differentiation][ad].

TensorFlow doesn't support [L-BFGS][l-bfgs] (which is what the original authors used), so this implementation uses [Adam][adam]. This may require some hyperparameter tuning to get optimal results.

## Running

This project uses the [uv](https://docs.astral.sh/uv/) project and package manager; you can install it by [following these instructions](https://docs.astral.sh/uv/getting-started/installation/) (e.g., installing it with your system package manager, like `brew install uv`).

You also need to download [required data files](#requirements).

After that, you can run this program with:

```bash
uv run neural_style.py --content <content file> --styles <style file> --output <output file>

Run uv run neural_style.py --help to see a list of all options.

Use --checkpoint-output and --checkpoint-iterations to save checkpoint images.

Use --iterations to change the number of iterations (default 1000). For a 512×512 pixel content file, 1000 iterations take about 90 seconds on a standard MacBook Pro and less on a more powerful GPU. 
```

Example 1
Running it for 500-2000 iterations produces nice results. Some images or output sizes may require tuning (especially --content-weight, --style-weight, and --learning-rate).

The following example was run for 1000 iterations to produce the result (with default parameters)


# Tweaking
--style-layer-weight-exp tweaks the abstraction level of style transfer. Lower values favor finer features; higher values favor coarser features. Default is 1.0.

--content-weight-blend controls how much content details are preserved (0.0 to 1.0).

--pooling selects pooling layers: either max or avg. Average pooling tends to produce smoother style transfer.

--preserve-colors applies a post-processing step to preserve colors from the original image.

## Requirements
Data Files
Pre-trained VGG network (SHA256 abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695) - put it in the top level of this repository, or specify its location with the --network option.
