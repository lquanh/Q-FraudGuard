# Giấy phép và ghi công của thành phần bên thứ ba

Giấy phép MIT của Q-FraudGuard áp dụng cho mã và tài liệu do dự án viết. Mẫu dữ liệu và lịch sử lưu trong mô hình có điều khoản tại danh mục dữ liệu. [Giấy phép Q-FraudGuard](LICENSE) · [Dữ liệu, mô hình và kết quả](data/licenses/LICENSES.md)

## Thư viện Python

Các gói được cài riêng qua `pip`. Giấy phép trong bảng lấy từ metadata của gói hoặc nguồn phát hành. Gói nhị phân có thể kèm thư viện với thông báo riêng, ví dụ OpenBLAS trong một số bản NumPy và SciPy hoặc thư viện NVIDIA khi dùng bản PyTorch hỗ trợ CUDA.

| Phạm vi | Thành phần | Giấy phép và nguồn |
|---|---|---|
| Chấm điểm và lưu mô hình | joblib, LightGBM | BSD-3-Clause [joblib](https://pypi.org/project/joblib/) · MIT [LightGBM](https://pypi.org/project/lightgbm/4.6.0/) |
| Dữ liệu và tính toán | NumPy, pandas, SciPy, scikit-learn, threadpoolctl | BSD-3-Clause [NumPy](https://github.com/numpy/numpy/blob/v2.1.3/LICENSE.txt) · [pandas](https://pypi.org/project/pandas/2.3.3/) · [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) · [scikit-learn](https://pypi.org/project/scikit-learn/1.8.0/) · [threadpoolctl](https://pypi.org/project/threadpoolctl/) |
| Dashboard | Streamlit | Apache-2.0 [Streamlit](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| Đồ thị theo thời gian | PyTorch | BSD-3-Clause [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) · [Thông báo thành phần đi kèm](https://github.com/pytorch/pytorch/blob/main/NOTICE) |
| API Core Banking | FastAPI, Pydantic, Uvicorn | MIT [FastAPI](https://pypi.org/project/fastapi/) · [Pydantic](https://pypi.org/project/pydantic/) · BSD-3-Clause [Uvicorn](https://pypi.org/project/uvicorn/) |
| Mô phỏng và tối ưu QUBO | OpenJij, Qiskit | Apache-2.0 [OpenJij](https://pypi.org/project/openjij/0.12.0/) · [Qiskit](https://pypi.org/project/qiskit/2.4.2/) |
| D-Wave | Ocean SDK, dimod, dwave-cloud-client, dwave-graphs, dwave-system, minorminer | Apache-2.0 cho các dự án chính [Ocean SDK](https://pypi.org/project/dwave-ocean-sdk/9.4.0/) · [dimod](https://pypi.org/project/dimod/) · [dwave-cloud-client](https://pypi.org/project/dwave-cloud-client/) · [dwave-graphs](https://pypi.org/project/dwave-graphs/) · [dwave-system](https://pypi.org/project/dwave-system/) · [minorminer](https://github.com/dwavesystems/minorminer/blob/main/LICENSE). Đối chiếu các tệp giấy phép kèm từng gói Ocean. |
| Đồ thị benchmark | NetworkX | BSD-3-Clause [NetworkX](https://pypi.org/project/networkx/) |
| Quafu | quarkstudio, quarkcircuit | MIT theo metadata của bản dùng trong benchmark [quarkstudio 7.3.9](https://pypi.org/project/quarkstudio/7.3.9/) · [quarkcircuit 0.5.13](https://pypi.org/project/quarkcircuit/0.5.13/) |
| Build, kiểm thử và tạo ảnh | Hatchling, pytest, Ruff, HTTPX, Pillow | MIT [Hatchling](https://pypi.org/project/hatchling/) · [pytest](https://pypi.org/project/pytest/) · [Ruff](https://pypi.org/project/ruff/) · BSD-3-Clause [HTTPX](https://pypi.org/project/httpx/) · MIT-CMU [Pillow](https://pypi.org/project/pillow/) |

## Dashboard và tài sản thiết kế

| Thành phần | Nơi sử dụng | Giấy phép và ghi công |
|---|---|---|
| Fira Code | Tải qua Google Fonts trong CSS và cockpit | OFL-1.1 · The Fira Code Project Authors. [Toàn văn](dashboard/design/LICENSE-FiraCode.txt) · [Nguồn](https://github.com/google/fonts/blob/main/ofl/firacode/OFL.txt) |
| Fira Sans | Tải qua Google Fonts trong CSS và cockpit | OFL-1.1 · The Mozilla Foundation và Telefonica S.A. [Toàn văn](dashboard/design/LICENSE-FiraSans.txt) · [Nguồn](https://github.com/google/fonts/blob/main/ofl/firasans/OFL.txt) |
| Material Symbols | Biểu tượng `:material:` do Streamlit cung cấp | Apache-2.0 · Google. [Giấy phép gốc](https://github.com/google/material-design-icons/blob/master/LICENSE) |
| Taste Skill | Hiệu ứng line engraving dùng để tạo `dashboard/assets/engraving.webp` | MIT · Leonxlnx. [Toàn văn](dashboard/design/LICENSE-taste-skill.txt) · [Nguồn](https://github.com/Leonxlnx/taste-skill/blob/main/LICENSE) |
| UI/UX Pro Max | Công cụ định dạng design token và CLI phát triển | MIT theo tệp `LICENSE` của dự án và metadata CLI · Next Level Builder. [Toàn văn](dashboard/design/LICENSE-ui-ux-pro-max.txt) · [Nguồn](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/blob/main/LICENSE) · [Phiên bản CLI](https://www.npmjs.com/package/ui-ux-pro-max-cli/v/2.15.0) |
| Morphicons | Nguồn tham khảo được ghi công trong mẫu cockpit | MIT · Guillermo. [Toàn văn](dashboard/design/LICENSE-morphicons.txt) · [Nguồn](https://github.com/guillermolg00/morphicons/blob/main/LICENSE) |

README trong gói `ui-ux-pro-max-cli@2.15.0` vẫn ghi `CC-BY-NC-4.0`. Giữ thông báo của từng thành phần upstream nếu phân phối mã hoặc tài sản đã lấy từ gói. Khóa npm ghi phiên bản và giấy phép của từng gói trực tiếp và bắc cầu. [Trao đổi tại upstream](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/issues/544) · [Khóa gói thiết kế](dashboard/design/package-lock.json)

## Dịch vụ QPU và các bản triển khai

- Quyền truy cập Quafu Haituo và D-Wave Leap theo điều khoản tài khoản của từng nhà cung cấp. Các gói Python có giấy phép riêng. D-Wave phân phối ứng dụng Problem Inspector `dwave-inspectorapp` theo EULA riêng nếu ứng dụng được cài thêm. [Quafu-SQC](https://quafu-sqc.baqis.ac.cn/) · [Thoả thuận D-Wave Leap](https://cloud.dwavesys.com/leap/signup/) · [EULA D-Wave](https://docs.dwavequantum.com/en/latest/licenses.html)
- Wheel `qf-guard` chứa package `qf_guard`, metadata và giấy phép MIT. Với bản phân phối chứa cả môi trường cài đặt, ví dụ ảnh container hoặc gói ngoại tuyến, lưu thông báo bản quyền, `LICENSE`, `NOTICE` và điều khoản của các gói và thư viện hệ thống thực sự được đóng gói. Danh sách phụ thuộc bắc cầu thay đổi theo hệ điều hành, phiên bản Python và tính năng được cài. Lưu kiểm kê từ chính môi trường của bản phân phối. [Định dạng `pip inspect`](https://pip.pypa.io/en/stable/reference/inspect-report/)

```powershell
python -m pip inspect > python-packages.json
npm --prefix dashboard/design ls --all --json > npm-packages.json
```

Chạy lệnh npm sau `npm --prefix dashboard/design ci` nếu bản dựng cần công cụ thiết kế. Dùng metadata và tệp giấy phép của đúng phiên bản cài đặt để đối chiếu. [Lệnh `npm ls`](https://docs.npmjs.com/cli/commands/npm-ls)
