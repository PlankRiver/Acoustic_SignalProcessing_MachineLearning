%% 批量绘制频谱图（MAT文件不动，图片保存到对应_plot文件夹）
clear; clc; close all;


% 设置为无显示模式，所有绘图操作在后台完成，不弹出窗口
set(0, 'DefaultFigureVisible', 'off');  
% ---------------------- 1. 固定根路径配置 ----------------------
root_dir = 'G:\东一千2月现场采集\下缆\数据集\频谱';  

% 检查根目录是否存在
if ~exist(root_dir, 'dir')
    error('根目录不存在，请检查路径：%s', root_dir);
end

% ---------------------- 2. 第一步：获取所有原始类别文件夹（排除_plot） ----------------------
original_folders = dir(fullfile(root_dir, '*'));
original_folders = original_folders([original_folders.isdir]);  % 筛选文件夹
original_folders = original_folders(~ismember({original_folders.name}, {'.', '..'}));  % 排除系统目录
original_folders = original_folders(~contains({original_folders.name}, '_plot'));  % 排除已有的_plot文件夹

if isempty(original_folders)
    error('根目录下未找到原始类别文件夹（如背景、拆卸等），请检查目录结构！');
end

% ---------------------- 3. 第二步：批量处理每个原始类别文件夹 ----------------------
for i = 1:length(original_folders)
    % 获取原始类别文件夹信息
    category_name = original_folders(i).name;  % 如：背景
    original_folder = fullfile(root_dir, category_name);  % 原始MAT文件所在文件夹
    plot_folder = fullfile(root_dir, [category_name, '_plot']);  % 图片保存的_plot文件夹
    
    % 自动创建_plot文件夹（不存在则创建）
    if ~exist(plot_folder, 'dir')
        mkdir(plot_folder);
        fprintf('已创建文件夹：%s\n', plot_folder);
    end
    
    % 查找原始文件夹内的所有.mat文件（MAT文件不动，直接读取）
    mat_files = dir(fullfile(original_folder, '*.mat'));
    if isempty(mat_files)
        warning('[%s] 文件夹中未找到.mat文件，跳过！', category_name);
        continue;
    end
    
    % 遍历每个.mat文件（直接读取原始文件夹中的MAT，无需移动）
    for j = 1:length(mat_files)
        mat_file_name = mat_files(j).name;
        mat_path = fullfile(original_folder, mat_file_name);
        fprintf('\n正在处理：%s/%s\n', category_name, mat_file_name);
        
        try
            % 加载原始文件夹中的MAT文件（无需移动）
            load(mat_path);  
            
            % 检查必要变量是否存在
            if ~exist('ROI_SF_STFT', 'var') || ~exist('Fspec', 'var') || ~exist('Tscaled', 'var')
                error('缺少必要变量（ROI_SF_STFT/Fspec/Tscaled）！');
            end
            
            % ---------------------- 4. 绘制频谱图（无分贝转换） ----------------------
            figure('Color','w', 'Position', [100 100 1000 600]);  % 图窗大小
            
            % 直接使用原始频谱数据（取绝对值避免复数显示问题）
            imagesc(Tscaled, Fspec, abs(ROI_SF_STFT), [0,1000]);  
            
            % 样式设置
            colormap('jet');  
            colorbar;         
            xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('STFT Spectrogram - %s', category_name), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            set(gca, 'YDir', 'normal');  % 频率轴从下到上递增
            grid on; box on;
            
            % ---------------------- 5. 保存图片到对应的_plot文件夹 ----------------------
            img_file_name = strrep(mat_file_name, '.mat', '.png');  % 图片名与MAT文件名一致
            img_save_path = fullfile(plot_folder, img_file_name);  % 保存到_plot文件夹
            
            % 保存高清PNG图片（300DPI）
            print(gcf, img_save_path, '-dpng', '-r300');  
            close(gcf);  % 关闭图窗，释放内存
            
            fprintf('已保存图片到_plot文件夹：%s\n', img_save_path);
            
        catch ME
            warning('处理 [%s/%s] 失败：%s', category_name, mat_file_name, ME.message);
            close(gcf);  % 即使出错也关闭图窗
            continue;
        end
    end
end

fprintf('\n===== 批量绘图完成 =====\n');
fprintf('1. MAT文件仍保留在原始类别文件夹中，未做任何移动；\n');
fprintf('2. 所有频谱图已保存到对应的_plot子文件夹中！\n');