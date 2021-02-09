# AODNet

[![](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/soumik12345/aodnet/app.py)

Tensorflow implementation of [An All-in-One Network for Dehazing and Beyond](https://arxiv.org/pdf/1707.06543.pdf).

![](./assets/aodnet_architecture.jpg)

**Training Notebook:** [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/AODNet/blob/master/notebooks/AODnet_Train.ipynb)

**Training Logs:** [https://wandb.ai/19soumik-rakshit96/aodnet](https://wandb.ai/19soumik-rakshit96/aodnet)

## Instructions for running Inference

- `python3 -m pip install -r requirements.txt`

- `python3 -m streamlit run app.py`

## Results

### NYU-2 Test Set

![](assets/test_set/pred_1.png)

![](assets/test_set/pred_2.png)

![](assets/test_set/pred_3.png)

![](assets/test_set/pred_4.png)

![](assets/test_set/pred_8.png)

![](assets/test_set/pred_10.png)

![](assets/test_set/pred_11.png)

### Inference on Real-world Hazy Images

<table>
    <thead>
        <td>
            Original Hazy Image
        </td>
        <td>
            Predicted Image
        </td>
    </thead>
    <tbody>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_1.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_1.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_2.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_2.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_3.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_3.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_4.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_4.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_5.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_5.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_6.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_6.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_7.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_7.jpeg"></td>
        </tr>
        <tr>
            <td><img src="assets/sample_test_images/sample_image_8.jpg"></td>
            <td><img src="assets/sample_pred_images/sample_pred_8.jpeg"></td>
        </tr>
    </tbody>
</table>

## Reference

```
@misc{
    1707.06543,
    Author = {Boyi Li and Xiulian Peng and Zhangyang Wang and Jizheng Xu and Dan Feng},
    Title = {An All-in-One Network for Dehazing and Beyond},
    Year = {2017},
    Eprint = {arXiv:1707.06543},
}
```
