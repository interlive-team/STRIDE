<div align="center">

# STRIDE

### When to Speak Meets Sequence Denoising for Streaming Video Understanding

[\[📜 Paper\]](https://arxiv.org/abs/2603.XXXXX)
[\[🌐 Project Page\]](https://interlive-team.github.io/STRIDE)
[\[🤗 Models\]](https://huggingface.co/interlive)

Junho Kim<sup>1*</sup>, Hosu Lee<sup>2*</sup>, James M. Rehg<sup>1</sup>, Minsu Kim<sup>3†</sup>, Yong Man Ro<sup>2†</sup>

<sup>1</sup>UIUC · <sup>2</sup>KAIST · <sup>3</sup>Google DeepMind

</div>

## Introduction

**STRIDE** (**S**tructured **T**emporal **R**efinement with **I**terative **DE**noising) is a lightweight proactive activation model for streaming video understanding. It employs a masked diffusion module at the activation interface to jointly predict and progressively refine activation signals over a sliding temporal window, producing temporally coherent proactive responses in online streaming scenarios.

## TODO

- [x] Paper release
- [x] Model weights release ([STRIDE-2B](https://huggingface.co/interlive/STRIDE-2B))
- [x] [Demo website](https://interlive-team.github.io/STRIDE)
- [ ] Training code
- [ ] Evaluation scripts

## Citation

```bibtex
@article{kim2026stride,
  title={STRIDE: When to Speak Meets Sequence Denoising for Streaming Video Understanding},
  author={Kim, Junho and Lee, Hosu and Rehg, James M. and Kim, Minsu and Ro, Yong Man},
  journal={arXiv preprint arXiv:2603.XXXXX},
  year={2026}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
