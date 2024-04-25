Kho lưu trữ này cung cấp một tập hợp các mô hình chuyển văn bản thành giọng nói (TTS) phổ biến rộng rãi trong TensorFlow Lite (TFLite). Các mô hình này chủ yếu đến từ hai kho lưu trữ - [TTS](https://github.com/mozilla/TTS) và [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS). Chúng tôi cung cấp Sổ ghi chép Colab toàn diện hiển thị quá trình suy luận và chuyển đổi mô hình bằng TFLite. Điều này bao gồm cả việc chuyển đổi các mô hình PyTorch sang TFLite.

TTS là một quy trình gồm hai bước - đầu tiên bạn tạo biểu đồ phổ MEL bằng mô hình TTS, sau đó chuyển nó tới VOCODER để tạo dạng sóng âm thanh. Chúng tôi bao gồm cả hai mô hình này trong kho lưu trữ này.

**Lưu ý** rằng các mô hình này được đào tạo trên [tập dữ liệu LJSpeech](https://www.tensorflow.org/datasets/catalog/ljspeech).

[Đây là kết quả mẫu](https://storage.googleapis.com/demo-experiments/demo_tts.wav) (với Fastspeech2 và MelGAN) cho văn bản “Bill có thói quen tự hỏi mình".

## Bao gồm các mô hình

- TTS:
    - [x] [Tacotron2](https://github.com/NVIDIA/tacotron2)
    - [x] [Fastspeech2](https://arxiv.org/abs/2006.04558)
    - [x] [Forward Tacotron](https://github.com/as-ideas/ForwardTacotron)
    - [ ] [Glow TTS](https://arxiv.org/abs/2005.11129)*
    - [ ] [Transformer TTS](https://arxiv.org/abs/1809.08895)
- VOCODER:
    - [x] [MelGAN](https://arxiv.org/abs/1910.06711)
    - [x] [Multi-Band MelGAN](https://arxiv.org/abs/2005.05106) (MB MelGAN)
    - [x] [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
    - [x] [HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf)

Trong tương lai, chúng tôi có thể bổ sung thêm nhiều mẫu mã hơn.

<small> *Hiện tại, việc chuyển đổi mẫu Glow TTS chưa khả dụng (tham khảo sự cố [tại đây](https://github.com/pytorch/pytorch/issues/50009)). </small>

Hiện tại, **Forward Tacotron** chỉ hỗ trợ Chuyển đổi ONNX. Đã xảy ra sự cố khi chuyển đổi sang Định dạng biểu đồ TensorFlow. (Tham khảo [vấn đề] này(https://github.com/onnx/onnx-tensorflow/issues/853) để biết thêm chi tiết).

**Ghi chú:**

- Dữ liệu huấn luyện được sử dụng cho HiFi-GAN (tạo phổ MEL) khác với các mô hình khác như Tacotron2, FastSpech2. Vì vậy, nó không tương thích với các kiến trúc khác có sẵn trong kho lưu trữ này.
- Nếu bạn muốn sử dụng HiFi-GAN trong kịch bản end-to-end, bạn có thể tham khảo [notebook] này (https://github.com/jaywalnut310/glow-tts/blob/master/inference_hifigan.ipynb). Trong tương lai, chúng tôi dự định làm cho nó tương thích với các kiến trúc khác và thêm nó vào [sổ ghi chép tổng thể](https://github.com/tulasiram58827/TTS_TFLite/blob/main/End_to_End_TTS.ipynb). Giữ nguyên!

## Về sổ tay
- `End_to_End_TTS.ipynb`: Sổ ghi chép này cho phép bạn tải lên các mô hình TTS và VOCODER khác nhau (được liệt kê ở trên) và thực hiện suy luận. 
- `MelGAN_TFLite.ipynb`: Hiển thị quá trình chuyển đổi mô hình của MelGAN.
- `Parallel_WaveGAN_TFLite.ipynb`: Hiển thị quá trình chuyển đổi mô hình của Parallel WaveGAN. 
- `HiFi-GAN.ipynb`: Hiển thị quá trình chuyển đổi mô hình của HiFi-GAN.
- `Forward_Tacotron_PyTorch_TFLite.ipynb` : Chuyển đổi mô hình Tacotron chuyển tiếp sang ONNX. Trong tương lai nó sẽ được cập nhật để hỗ trợ chuyển đổi TFLite.

Quy trình chuyển đổi mô hình cho Tacotron2, Fastspeech2 và Multi-Band MelGAN có sẵn qua các sổ ghi chép sau:

- [Tacotron2 & Multi-Band MelGAN](https://colab.research.google.com/github/mozilla/TTS/blob/master/notebooks/DDC_TTS_and_MultiBand_MelGAN_TFLite_Example.ipynb)
- [Fastspeech2](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/notebooks/TensorFlowTTS_FastSpeech_with_TFLite.ipynb)
## Điểm chuẩn mô hình

Sau khi chuyển đổi sang TFLite, chúng tôi đã sử dụng [Công cụ đo điểm chuẩn](https://www.tensorflow.org/lite/performance/measurement) để báo cáo các chỉ số hiệu suất của nhiều mô hình như độ trễ suy luận, mức sử dụng bộ nhớ cao nhất. Chúng tôi đã sử dụng Redmi K20 cho mục đích này. Đối với tất cả các thử nghiệm, chúng tôi giữ số lượng luồng ở mức một và chúng tôi sử dụng CPU của Redmi K20 chứ không sử dụng bộ tăng tốc phần cứng nào khác.

| **Mô hình**        | **Lượng tử hóa** | **Kích thước mô hình (MB)** | **Độ trễ suy luận trung bình (sec)** | **Mức chiếm dụng bộ nhớ (MB)** |
| ---------------- | ---------------- | :-----------------: | :----------------------------------:| :-----------------------: |
| Parallel WaveGAN | Dynamic-range    | 5.7                 | 0.04                                | 31.5                      |
| Parallel WaveGAN | Float16          | 3.2                 | 0.05                                | 34                        |
| MelGAN           | Dynamic-range    | 17                  | 0.51                                | 81                        |
| MelGAN           | Float16          | 8.3                 | 0.52                                | 89                        |
| MB MelGAN        | Dynamic-range    | 17                  | 0.02                                | 17                        |
| HiFi-GAN         | Dynamic-range    | 3.5                 | 0.0015                              | 9.88                      |
| HiFi-GAN         | Float16          | 2.9                 | 0.0036                              | 20.3                      | 
| Tacotron2        | Dynamic-range    | 30.1                | 1.66                                | 75                        |
| Fastspeech2      | Dynamic-range    | 30                  | 0.11                                | 55                        |

**Ghi chú**:

- Tất cả các mô hình trên đều hỗ trợ đầu vào có hình dạng động. Tuy nhiên, việc đo điểm chuẩn cho các mô hình MelGAN kích thước đầu vào động hiện không được hỗ trợ. Vì vậy, để đánh giá các mô hình đó, chúng tôi đã sử dụng đầu vào có hình dạng (100, 80).
- Tương tự đối với mô hình kích thước đầu vào động đo điểm chuẩn Fastspeech2 bị lỗi. Vì vậy, để đo điểm chuẩn, chúng tôi đã sử dụng các đầu vào có hình dạng (1, 50) trong đó 50 đại diện cho số lượng mã thông báo. [Chuỗi vấn đề này](https://github.com/tensorflow/tensorflow/issues/45986) cung cấp thêm thông tin chi tiết.

## 🔈 Âm thanh mẫu

Tất cả sự kết hợp của các mẫu đều có sẵn trong thư mục `audio_samples`. Để nghe trực tiếp mà không cần tải xuống, hãy tham khảo thư mục [Sound Cloud](https://soundcloud.com/tulasi-ram-887761209) này.

## Người giới thiệu
- [Lượng tử hóa phạm vi động trong TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Lượng tử hóa Float16 trong TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_float16_quant)
