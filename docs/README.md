# Session Template
```
cv::Mat im0 = cv::imread("/path/to/image");
cv::Mat im1 = cv::imread("/path/to/image");
cv::Mat disparity, output;
retinify::Session session(RETINIFY_DEFALUT_MODEL_PATH);
session.UploadInputData(0, im0);
session.UploadInputData(1, im1);
session.Run();
session.DownloadOutputData(0, disparity);
output = retinify::ColoringDisparity(disparity, 120);
cv::imwrite("/path/to/output", output);
```