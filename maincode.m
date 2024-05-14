% Get all image files in the dataset folder
image_files = dir('dataset/*.jpg');

% Check for the output_folder directory for histogram equalization process and delete if exists
if exist('output_folder', 'dir')
    rmdir('output_folder', 's'); % 's' option removes subfolders as well
end

% Create a new folder for histogram equalization process
mkdir('output_folder');

% Load the first reference images and calculate their histograms
reference_img_1 = imread("dataset/TRAIN101104.jpg");
[counts_1, bins_1] = imhist(reference_img_1);

reference_img_2 = imread("dataset/TRAIN101210.jpg");
[counts_2, ~] = imhist(reference_img_2);

% Calculate the mean of the histograms
mean_histogram = (counts_1 + counts_2) / 2;

% Loop through all image files and apply the histeq operation
for i = 1:length(image_files)
    % Load the image
    img = imread(fullfile('dataset', image_files(i).name));
    
    % Apply the histeq operation
    img_histeq = histeq(img, mean_histogram);
    
    % Generate the new filename and save the histeq processed image
    [~, filename, ext] = fileparts(image_files(i).name);
    output_filename = fullfile('output_folder', [filename '_histeq' ext]);
    imwrite(img_histeq, output_filename);
end

% Create a figure to plot the histograms
figure;

% Plot the mean histogram
plot(bins_1, mean_histogram, 'k', 'LineWidth', 1.5);

% Calculate histograms of adjustable and adjusted images
adjustable_img = imread("dataset/TRAIN100952.jpg");
[counts_adjustable, bins_adjustable] = imhist(adjustable_img);

adjusted_img = imread("output_folder/TRAIN100952_histeq.jpg");
[counts_adjusted, bins_adjusted] = imhist(adjusted_img);

% Plot the histograms of adjustable and adjusted images
hold on;
plot(bins_adjustable, counts_adjustable, 'r--', 'LineWidth', 1.5);
plot(bins_adjusted, counts_adjusted, 'g--', 'LineWidth', 1.5);
hold off;

% Edit the graph
title('Histogram Comparison of Reference, Adjustable, and Adjusted Images');
xlabel('Pixel Value');
ylabel('Frequency');
legend('Mean Reference Histogram', 'Adjustable Image', 'Adjusted Image');
grid on;

% Zoom in on the graph
xlim([0 100]);
ylim([0 (0.5)*10^6]);





% Check for the output_gray_folder directory for grayscale images and delete if exists
if exist('output_gray_folder', 'dir')
    rmdir('output_gray_folder', 's'); % 's' option removes subfolders as well
end

% Define the name of the output gray folder
output_gray_folder = 'output_gray_folder';

% Create a new folder for grayscale images
mkdir(output_gray_folder);

% Get all image files in the output_folder directory
image_files = dir('output_folder/*.jpg');


% Loop through all image files and convert them to grayscale
for i = 1:length(image_files)
    % Load the image
    img = imread(fullfile('output_folder', image_files(i).name));
    
    % Convert the image to grayscale
    img_gray = im2gray(img);
    
    % Generate the filename for the grayscale image to be saved
    [~, filename, ext] = fileparts(image_files(i).name);
    output_gray_filename = fullfile(output_gray_folder, [filename '_gray' ext]);
    
    % Save the grayscale image
    imwrite(img_gray, output_gray_filename);
end








%GAUSSIAN FILTER
% Filtreleme işlemi için output_folder klasörünü kontrol edin ve varsa silin
if exist('output_filtered_folder', 'dir')
    rmdir('output_filtered_folder', 's'); % 's' seçeneği alt klasörleri de siler
end

% Filtreleme işlemi için yeni klasörü oluşturun
mkdir('output_filtered_folder');

% Define parameters for the Gaussian filter
sigma = 10; % Standard deviation
kernel_size = 2 * ceil(3 * sigma) + 1; % Kernel size

% Create the Gaussian filter kernel
gaussian_kernel = fspecial('gaussian', [kernel_size kernel_size], sigma);

% Load grayscale images from the folder
image_files_gray = dir('output_gray_folder/*.jpg');

% Loop through each grayscale image
for i = 1:length(image_files_gray)
    % Read the grayscale image
    img_gray = imread(fullfile('output_gray_folder', image_files_gray(i).name));
    
    % Filter the grayscale image using the Gaussian filter
    filtered_image = imfilter(img_gray, gaussian_kernel, 'conv', 'replicate');
    
    % Save the filtered image
    [~, filename, ext] = fileparts(image_files_gray(i).name);
    output_filtered_filename = fullfile('output_filtered_folder', [filename '_filtered' ext]);
    imwrite(filtered_image, output_filtered_filename);
end


% Show kernel of the Gaussian filter in 3D
figure;
surf(gaussian_kernel);
title('3D Visualization of Gaussian Filter Kernel');
xlabel('Column');
ylabel('Row');
zlabel('Value');


% Calculate the FFT2 of the kernel
fft_kernel = fftshift(fft2(gaussian_kernel));

% Calculate the FFT2 of the resized original image
fft_image = fftshift(fft2(double(img_gray)));

% Calculate the FFT2 of the image after Gaussian filtering
fft_filtered_image = fftshift(fft2(double(filtered_image)));

% Display the FFT results as 2D images
figure;
subplot(1, 3, 1);
imagesc(log(abs(fft_kernel) + 1));
title('FFT of Kernel (2D Image)');
axis square;
subplot(1, 3, 2);
imagesc(log(abs(fft_image) + 1));
title('FFT of Original Image (2D Image)');
axis square;
subplot(1, 3, 3);
imagesc(log(abs(fft_filtered_image) + 1));
title('FFT of Filtered Image (2D Image)');
axis square;
colormap(gca, 'jet');
colorbar;


% Calculate 1D FFT of the original image along rows
fft_image_row = fftshift(fft(double(img_gray), [], 2), 2);

% Calculate 1D FFT of the filtered image along rows
fft_filtered_row = fftshift(fft(double(filtered_image), [], 2), 2);

% Calculate magnitudes
magnitude_fft_image_row = abs(fft_image_row);
magnitude_fft_filtered_row = abs(fft_filtered_row);

% Display magnitude spectrum in logarithmic scale
figure;
plot(1:size(magnitude_fft_image_row, 2), log(magnitude_fft_image_row(ceil(end/2), :)), 'b');
hold on;
plot(1:size(magnitude_fft_filtered_row, 2), log(magnitude_fft_filtered_row(ceil(end/2), :)), 'k');
title('Magnitude Spectrum along Rows (1D)');
xlabel('Frequency');
ylabel('Magnitude (log scale)');
legend('Before Filtration', 'After Filtration');

% Calculate 1D FFT of the Gaussian filter kernel along rows
fft_kernel_row = fftshift(fft(gaussian_kernel(ceil(end/2), :)));

% Calculate magnitudes
magnitude_fft_kernel_row = abs(fft_kernel_row);

% Display magnitude spectrum of the Gaussian filter kernel in logarithmic scale
figure;
plot(1:length(magnitude_fft_kernel_row), log(magnitude_fft_kernel_row), 'r');
title('Magnitude Spectrum of Gaussian Filter Kernel along Rows (1D)');
xlabel('Frequency');
ylabel('Magnitude (log scale)');









%LOG FILTER
% Define the output folder for filtered images
output_filtered_log_folder = 'output_filtered_log_folder';

% Filtreleme işlemi için output_folder klasörünü kontrol edin ve varsa silin
if exist('output_filtered_log_folder', 'dir')
    rmdir('output_filtered_log_folder', 's'); % 's' seçeneği alt klasörleri de siler
end

% Filtreleme işlemi için yeni klasörü oluşturun
mkdir('output_filtered_log_folder');


% Define parameters for the LoG filter
sigma_log = 3; % Standard deviation for Gaussian smoothing
filter_size = 2 * ceil(3 * sigma_log) + 1; % Filter size

% Create the LoG filter kernel
log_kernel = fspecial('log', filter_size, sigma_log);


% Load grayscale images from the folder
image_files_gray = dir('output_gray_folder/*.jpg');

% Loop through each grayscale image
for i = 1:length(image_files_gray)
    % Read the grayscale image
    img_gray = imread(fullfile('output_gray_folder', image_files_gray(i).name));
    
    % Filter the grayscale image using the LoG filter
    filtered_image_log = imfilter(double(img_gray), log_kernel, 'conv', 'replicate');
    
    % Save the filtered image
    [~, filename, ext] = fileparts(image_files_gray(i).name);
    output_filtered_log_filename = fullfile(output_filtered_log_folder, [filename '_filtered_log' ext]);
    imwrite(filtered_image_log, output_filtered_log_filename);
end

% Display the LoG filter kernel
figure;
surf(log_kernel);
title('3D Visualization of LoG Filter Kernel');
xlabel('Column');
ylabel('Row');
zlabel('Value');


% Calculate the FFT2 of the kernel
fft_kernel = fftshift(fft2(log_kernel));

% Calculate the FFT2 of the resized original image
fft_image = fftshift(fft2(double(img_gray)));

% Calculate the FFT2 of the image after Gaussian filtering
fft_filtered_image = fftshift(fft2(double(filtered_image_log)));

% Display the FFT results as 2D images
figure;
subplot(1, 3, 1);
imagesc(log(abs(fft_kernel) + 1));
title('FFT of Kernel (2D Image)');
axis square;
subplot(1, 3, 2);
imagesc(log(abs(fft_image) + 1));
title('FFT of Original Image (2D Image)');
axis square;
subplot(1, 3, 3);
imagesc(log(abs(fft_filtered_image) + 1));
title('FFT of Filtered Image (2D Image)');
axis square;
colormap(gca, 'jet');
colorbar;


% Calculate 1D FFT of the original image along rows
fft_image_row = fftshift(fft(double(img_gray), [], 2), 2);

% Calculate 1D FFT of the filtered image along rows
fft_filtered_row = fftshift(fft(double(filtered_image_log), [], 2), 2);

% Calculate magnitudes
magnitude_fft_image_row = abs(fft_image_row);
magnitude_fft_filtered_row = abs(fft_filtered_row);

% Display magnitude spectrum in logarithmic scale
figure;
plot(1:size(magnitude_fft_image_row, 2), log(magnitude_fft_image_row(ceil(end/2), :)), 'b');
hold on;
plot(1:size(magnitude_fft_filtered_row, 2), log(magnitude_fft_filtered_row(ceil(end/2), :)), 'k');
title('Magnitude Spectrum along Rows (1D)');
xlabel('Frequency');
ylabel('Magnitude (log scale)');
legend('Before Filtration', 'After Filtration');

% Calculate 1D FFT of the LoG filter kernel along rows
fft_kernel_row_log = fftshift(fft(log_kernel(ceil(end/2), :)));

% Calculate magnitudes
magnitude_fft_kernel_row_log = abs(fft_kernel_row_log);

% Display magnitude spectrum of the LoG filter kernel in logarithmic scale
figure;
plot(1:length(magnitude_fft_kernel_row_log), log(magnitude_fft_kernel_row_log), 'r');
title('Magnitude Spectrum of LoG Filter Kernel along Rows (1D)');
xlabel('Frequency');
ylabel('Magnitude (log scale)');






% Filtrelenmiş resimler için klasörü tanımlayın
output_doublefiltered_log_folder = 'output_doublefiltered_log_folder';

% Filtreleme klasörünü kontrol edin ve varsa silin
if exist(output_doublefiltered_log_folder, 'dir')
    rmdir(output_doublefiltered_log_folder, 's'); % 's' seçeneği alt klasörleri de siler
end

% Filtreleme klasörünü oluşturun
mkdir(output_doublefiltered_log_folder);


% LoG filtresi için parametreleri tanımlayın
sigma_log = 2; % Gaussian düzgünleştirme için standart sapma
filter_size = 2 * ceil(3 * sigma_log) + 1; % Filtre boyutu

% LoG filtresi çekirdeğini oluşturun
log_kernel = fspecial('log', filter_size, sigma_log);


% Klasörden filtrelenmiş resimleri yükleyin
image_files_filtered = dir('output_filtered_folder/*.jpg');

% Her bir resim için işlem yapın
for i = 1:length(image_files_filtered)
    % Filtrelenmiş resmi okuyun
    img_filtered = imread(fullfile('output_filtered_folder', image_files_filtered(i).name));
    
    % LoG filtresi kullanarak resmi tekrar filtreleyin
    doublefiltered_image_log = imfilter(double(img_filtered), log_kernel, 'conv', 'replicate');
    
    % Filtrelenmiş resmi kaydedin
    [~, filename, ext] = fileparts(image_files_filtered(i).name);
    output_doublefiltered_log_filename = fullfile(output_doublefiltered_log_folder, [filename '_doublefiltered_log' ext]);
    imwrite(doublefiltered_image_log, output_doublefiltered_log_filename);
end







% Define the output folder for segmented images
output_segmented_folder = 'segmented_images';

% Klasörün varlığını kontrol edin ve varsa silin
if exist(output_segmented_folder, 'dir')
    rmdir(output_segmented_folder, 's'); % Alt klasörleri de siler
end

% Yeni klasörü oluşturun
mkdir(output_segmented_folder);

% Get a list of all image files in the input folder
image_files = dir(fullfile(output_filtered_log_folder, '*.jpg'));

% Loop through each image file
for i = 1:numel(image_files)
    % Read the image
    img = imread(fullfile(output_filtered_log_folder, image_files(i).name));
    
    % Apply segmentation using the segmentImage function
    [BW, maskedImage] = segmentImage(img);
    
    % Save the segmented image
    output_filename = fullfile(output_segmented_folder, [image_files(i).name(1:end-4) '_segmented.jpg']);
    imwrite(maskedImage, output_filename);
end






input_folder = 'segmented_images';
output_folder = 'segmented_images_processed';
segmentAndSaveImages(input_folder, output_folder);

function segmentAndSaveImages(input_folder, output_folder)
    % Define the output folder for segmented images
    
    % Klasörün varlığını kontrol edin ve varsa silin
    if exist(output_folder, 'dir')
        rmdir(output_folder, 's'); % Alt klasörleri de siler
    end

    % Yeni klasörü oluşturun
    mkdir(output_folder);

    % Get a list of all image files in the input folder
    image_files = dir(fullfile(input_folder, '*.jpg'));

    % Loop through each image file
    for i = 1:numel(image_files)
        % Read the image
        img = imread(fullfile(input_folder, image_files(i).name));

        % Apply segmentation using the filterRegions function
        [BW_out, ~] = filterRegions(img);

        % Save the segmented image
        output_filename = fullfile(output_folder, [image_files(i).name(1:end-4) '_segmented.jpg']);
        imwrite(BW_out, output_filename);
    end
end




