# LectureSync


[![version-image]][release-url]
[![release-date-image]][release-url]
[![license-image]][license-url]
[![codecov][codecov-image]][codecov-url]
[![jupyter-book-image]][docs-url]

<!-- Links: -->
[codecov-image]: https://codecov.io/gh/ypilseong/LectureSync/branch/main/graph/badge.svg?token=[REPLACE_ME]
[codecov-url]: https://codecov.io/gh/ypilseong/LectureSync
[pypi-image]: https://img.shields.io/pypi/v/LectureSync
[license-image]: https://img.shields.io/github/license/ypilseong/LectureSync
[license-url]: https://github.com/ypilseong/LectureSync/blob/main/LICENSE
[version-image]: https://img.shields.io/github/v/release/ypilseong/LectureSync?sort=semver
[release-date-image]: https://img.shields.io/github/release-date/ypilseong/LectureSync
[release-url]: https://github.com/ypilseong/LectureSync/releases
[jupyter-book-image]: https://jupyterbook.org/en/stable/_images/badge.svg

[repo-url]: https://github.com/ypilseong/LectureSync
[pypi-url]: https://pypi.org/project/LectureSync
[docs-url]: https://ypilseong.github.io/LectureSync
[changelog]: https://github.com/ypilseong/LectureSync/blob/main/CHANGELOG.md
[contributing guidelines]: https://github.com/ypilseong/LectureSync/blob/main/CONTRIBUTING.md
<!-- Links: -->

LectureSync transforms lecture recordings into summarized text and provides an interactive Q&A system to enhance learning efficiency.

- Documentation: [https://ypilseong.github.io/LectureSync][docs-url]
- GitHub: [https://github.com/ypilseong/LectureSync][repo-url]


LectureSync is an advanced platform that converts lecture recordings into summarized text and offers an interactive Q&A system powered by state-of-the-art speech-to-text and natural language processing technologies, enabling users to quickly access and comprehend key information while facilitating deeper engagement with educational content.




## Quick Start

To quickly get started with LectureSync, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ypilseong/LectureSync.git
    ```
2. Navigate to the project directory:
    ```bash
    cd LectureSync
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```bash
    streamlit run src/lecturesync/langchain/streamlit_frontend.py
    ```

> **Note:** Ensure you update the model configuration in `streamlit_frontend.py` to match your desired settings.
> **note**Example(Using Ollama Custom Model)
> If you want to use a custom Ollama model, modify the following line in `streamlit_frontend.py`:
> ```python
> model_url = 'Ollama URL'
> model_name = 'Your custom Model'
> ```



## Changelog

See the [CHANGELOG] for more information.

## Contributing

Contributions are welcome! Please see the [contributing guidelines] for more information.

## License

This project is released under the [MIT License][license-url].
