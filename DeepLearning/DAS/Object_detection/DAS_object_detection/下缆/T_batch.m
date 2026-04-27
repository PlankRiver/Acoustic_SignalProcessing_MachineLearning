%% ===================== 全局参数配置区 (可根据需求修改) =====================
close all; clear all; clc;
warning off; % 关闭无关警告

% 1. 批量处理路径配置
% 支持外部传入 csvRootFolder（分类根目录），否则使用默认路径
if ~exist('csvRootFolder','var') || isempty(csvRootFolder)
    csvRootFolder = 'E:\东一千2月现场采集\下缆'; % 根目录：其下分类文件夹的 csv 子目录才存放CSV
end
saveRoot = pwd; % 原始保存根路径(默认当前脚本路径)

% 探测根目录下的"分类/ csv"子目录，优先处理这些目录；如未找到，则按原有 csvRootFolder 直接处理
candidateCsvDirs = {};
subdirs = dir(csvRootFolder);
subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.','..'}));
for si = 1:numel(subdirs)
    catDir = fullfile(subdirs(si).folder, subdirs(si).name);
    csvDir = fullfile(catDir, 'csv');
    if isfolder(csvDir) && ~isempty(dir(fullfile(csvDir, '*.csv')))
        candidateCsvDirs{end+1} = csvDir; %#ok<AGROW>
    end
end
if isempty(candidateCsvDirs)
    candidateCsvDirs = {csvRootFolder}; % 兼容老用法：直接传入具体CSV目录
end

% 2. 信号处理参数 (修复LPF1多余逗号的语法错误)
Fs1 = 4000; % 原始采样率 4kHz
M = 8;      % 下采样倍数 8倍
Fs2 = Fs1/M;% 下采样后采样率 500Hz
% 50Hz通带/100Hz阻带(-80dB)的低通滤波器系数(4kHz采样)【修复：删掉多余的逗号】
LPF0=[0.000117481336437156,6.90025024939821e-05,8.82739294343904e-05,0.000110285772207978,0.000135115128030931,0.000162786935107966,0.000193235341874338,0.000226318247143453,0.000261885467656352,0.000299761410511061,0.000339602413624077,0.000380898320884262,0.000423214900743892,0.000466072599434334,0.000508548765055754,0.000550169844575579,0.000589892943037661,0.000626864490826878,0.000660046507844421,0.000688371337595354,0.000710680766312748,0.000725778445548169,0.000732454469681618,0.000729449072800322,0.000715480104360555,0.000689324487808824,0.000649757052644477,0.000595560556979922,0.000525704146388667,0.000439120362500214,0.000334961406566385,0.000212462023240829,7.10558696402076e-05,-8.96644488058920e-05,-0.000269885882565640,-0.000469558231733130,-0.000688398491157916,-0.000925851925709555,-0.00118105876669901,-0.00145288613670228,-0.00173989132076899,-0.00204028459576093,-0.00235201108183923,-0.00267264929922544,-0.00299950019148757,-0.00332953115725214,-0.00365943804993650,-0.00398562250140396,-0.00430422844592797,-0.00461117726903520,-0.00490217987250199,-0.00517277203523981,-0.00541837332237107,-0.00563430378657675,-0.00581582852163706,-0.00595823396481685,-0.00605682938903205,-0.00610704417712307,-0.00610443974697491,-0.00604480119614265,-0.00592415666556035,-0.00573885222187891,-0.00548560311872644,-0.00516153240504994,-0.00476422603268730,-0.00429178457798259,-0.00374284936629387,-0.00311664553035718,-0.00241302439818505,-0.00163246789619280,-0.000776134027674070,0.000154152152812243,0.00115586456717500,0.00222580299444723,0.00336007699156459,0.00455412696176012,0.00580274992234898,0.00710012013683992,0.00843982358208174,0.00981490669160938,0.0112179210727024,0.0126409725422137,0.0140757943243467,0.0155137988881277,0.0169461607022568,0.0183638739785871,0.0197578496035768,0.0211189745306948,0.0224382025416280,0.0237066357284760,0.0249156008517101,0.0260567302597258,0.0271220424197744,0.0281040165806873,0.0289956567023947,0.0297905696096547,0.0304830150343042,0.0310679687365855,0.0315411613868112,0.0318991337375304,0.0321392551925840,0.0322597591692132,0.0322597591692132,0.0321392551925840,0.0318991337375304,0.0315411613868112,0.0310679687365855,0.0304830150343042,0.0297905696096547,0.0289956567023947,0.0281040165806873,0.0271220424197744,0.0260567302597258,0.0249156008517101,0.0237066357284760,0.0224382025416280,0.0211189745306948,0.0197578496035768,0.0183638739785871,0.0169461607022568,0.0155137988881277,0.0140757943243467,0.0126409725422137,0.0112179210727024,0.00981490669160938,0.00843982358208174,0.00710012013683992,0.00580274992234898,0.00455412696176012,0.00336007699156459,0.00222580299444723,0.00115586456717500,0.000154152152812243,-0.000776134027674070,-0.00163246789619280,-0.00241302439818505,-0.00311664553035718,-0.00374284936629387,-0.00429178457798259,-0.00476422603268730,-0.00516153240504994,-0.00548560311872644,-0.00573885222187891,-0.00592415666556035,-0.00604480119614265,-0.00610443974697491,-0.00610704417712307,-0.00605682938903205,-0.00595823396481685,-0.00581582852163706,-0.00563430378657675,-0.00541837332237107,-0.00517277203523981,-0.00490217987250199,-0.00461117726903520,-0.00430422844592797,-0.00398562250140396,-0.00365943804993650,-0.00332953115725214,-0.00299950019148757,-0.00267264929922544,-0.00235201108183923,-0.00204028459576093,-0.00173989132076899,-0.00145288613670228,-0.00118105876669901,-0.000925851925709555,-0.000688398491157916,-0.000469558231733130,-0.000269885882565640,-8.96644488058920e-05,7.10558696402076e-05,0.000212462023240829,0.000334961406566385,0.000439120362500214,0.000525704146388667,0.000595560556979922,0.000649757052644477,0.000689324487808824,0.000715480104360555,0.000729449072800322,0.000732454469681618,0.000725778445548169,0.000710680766312748,0.000688371337595354,0.000660046507844421,0.000626864490826878,0.000589892943037661,0.000550169844575579,0.000508548765055754,0.000466072599434334,0.000423214900743892,0.000380898320884262,0.000339602413624077,0.000299761410511061,0.000261885467656352,0.000226318247143453,0.000193235341874338,0.000162786935107966,0.000135115128030931,0.000110285772207978,8.82739294343904e-05,6.90025024939821e-05,0.000117481336437156];
LPF1=[0.000098660554703	0.000098919543309	0.000145620936554	0.000204244408761	0.000275842158885	0.000361223339479	0.000460916382258	0.000574903738906	0.000702703643065	0.000843434928080	0.000995224401674	0.001155877170511	0.001322318161021	0.001490913286104	0.001657316304741	0.001816655822026	0.001963530031581	0.002092208265067	0.002196813247228	0.002271354176457	0.002310082963242	0.002307590225696	0.002259134041535	0.002160801923364	0.002009800906667	0.001804591425061	0.001545116132486	0.001232973515751	0.000871486076191	0.000465822809803	0.000022953825543	-0.000448365104131	-0.000937756259344	-0.001433356836913	-0.001922122359007	-0.002390118288211	-0.002822892600867	-0.003205921749422	-0.003525054681064	-0.003767029667343	-0.003919954809953	-0.003973832643049	-0.003920990633995	-0.003756545754162	-0.003478760535503	-0.003089345351224	-0.002593669143167	-0.002000847034982	-0.001323726889488	-0.000578728446603	0.000214406408039	0.001033044116304	0.001852159649276	0.002644990962030	0.003383775058698	0.004040577091870	0.004588181645518	0.005001019248424	0.005256114045376	0.005333981211353	0.005219514427652	0.004902753878243	0.004379575504426	0.003652222125204	0.002729684850089	0.001627889630141	0.000369690586687	-0.001015338609843	-0.002491349007914	-0.004016842910979	-0.005545533172806	-0.007027364587607	-0.008409718568951	-0.009638734943960	-0.010660752224835	-0.011423802319647	-0.011879124648348	-0.011982674876526	-0.011696546923487	-0.010990306818631	-0.009842153113003	-0.008239903694887	-0.006181733812300	-0.003676678612670	-0.000744848276733	0.002582647327925	0.006264062062475	0.010247687253112	0.014472794800610	0.018870861633728	0.023367009714721	0.027881669469176	0.032332380851827	0.036635718211661	0.040709271943874	0.044473632364959	0.047854333211545	0.050783682849858	0.053202456985000	0.055061369577077	0.056322325323329	0.056959373998451	0.056959373998451	0.056322325323329	0.055061369577077	0.053202456985000	0.050783682849858	0.047854333211545	0.044473632364959	0.040709271943874	0.036635718211661	0.032332380851827	0.027881669469176	0.023367009714721	0.018870861633728	0.014472794800610	0.010247687253112	0.006264062062475	0.002582647327925	-0.000744848276733	-0.003676678612670	-0.006181733812300	-0.008239903694887	-0.009842153113003	-0.010990306818631	-0.011696546923487	-0.011982674876526	-0.011879124648348	-0.011423802319647	-0.010660752224835	-0.009638734943960	-0.008409718568951	-0.007027364587607	-0.005545533172806	-0.004016842910979	-0.002491349007914	-0.001015338609843	0.000369690586687	0.001627889630141	0.002729684850089	0.003652222125204	0.004379575504426	0.004902753878243	0.005219514427652	0.005333981211353	0.005256114045376	0.005001019248424	0.004588181645518	0.004040577091870	0.003383775058698	0.002644990962030	0.001852159649276	0.001033044116304	0.000214406408039	-0.000578728446603	-0.001323726889488	-0.002000847034982	-0.002593669143167	-0.003089345351224	-0.003478760535503	-0.003756545754162	-0.003920990633995	-0.003973832643049	-0.003919954809953	-0.003767029667343	-0.003525054681064	-0.003205921749422	-0.002822892600867	-0.002390118288211	-0.001922122359007	-0.001433356836913	-0.000937756259344	-0.000448365104131	0.000022953825543	0.000465822809803	0.000871486076191	0.001232973515751	0.001545116132486	0.001804591425061	0.002009800906667	0.002160801923364	0.002259134041535	0.002307590225696	0.002310082963242	0.002271354176457	0.002196813247228	0.002092208265067	0.001963530031581	0.001816655822026	0.001657316304741	0.001490913286104	0.001322318161021	0.001155877170511	0.000995224401674	0.000843434928080	0.000702703643065	0.000574903738906	0.000460916382258	0.000361223339479	0.000275842158885	0.000204244408761	0.000145620936554	0.000098919543309	0.000098660554703];
LPF2 = ones(501,1)/501; % 功率积分滤波器

% 3. ROI提取参数 (与原脚本一致，无需修改)
DW = 2;       % 距离方向ROI半宽
TW = 1500;    % 时间方向ROI半宽
Tstep = 50;   % 时间方向搜索步长
Th = 3;       % 事件峰值触发阈值
FigSwitch = 0;% 绘图开关 0=关闭(批量推荐) 1=开启(单文件调试)

%% ===================== 批量遍历候选CSV目录 =====================
for folderIdx = 1:numel(candidateCsvDirs)
    curCsvDir = candidateCsvDirs{folderIdx};
    % 保存目录使用"分类"名称（csv 所在目录的上一级）
    [parentDir, ~] = fileparts(curCsvDir);
    [~, categoryName] = fileparts(parentDir);
    newSaveRoot = fullfile(saveRoot, categoryName);

    % 若分类目录下已存在 ROI_Data/ROI_data，则视为已处理，直接跳过
    roiDirCandidates = {fullfile(parentDir, 'ROI_Data'), fullfile(parentDir, 'ROI_data')};
    roiDirHit = '';
    for rdi = 1:numel(roiDirCandidates)
        if isfolder(roiDirCandidates{rdi})
            roiDirHit = roiDirCandidates{rdi};
            break;
        end
    end
    if ~isempty(roiDirHit)
        fprintf('跳过分类 %s：检测到 ROI_Data 目录：%s\n', categoryName, roiDirHit);
        continue;
    end

    % 若已处理过且 ROI 文件数 > 原始文件数，则跳过该类别
    [skipCat, roiCount, origCount, usedRoiDir] = shouldSkipCategory(parentDir, newSaveRoot);
    if skipCat
        fprintf('跳过分类 %s：检测到 ROI_Data 已处理（ROI文件数 %d > 原始文件数 %d），路径：%s\n', ...
            categoryName, roiCount, origCount, usedRoiDir);
        continue;
    end

    % 自动创建顶层文件夹（基于当前分类保存根路径）
    mkdirIfNotExist(fullfile(newSaveRoot, 'Feature_Image'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Image'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Data'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Data', 'ROI_SP'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Data', 'ROI_SF'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Data', 'DownSampling'))
    mkdirIfNotExist(fullfile(newSaveRoot, 'ROI_Data', 'ROI_SP_im'))

    %% ===================== 批量遍历CSV文件 =====================
    csvFiles = dir(fullfile(curCsvDir, '*.csv'));
    fileNum = length(csvFiles);
    if fileNum == 0
        fprintf('跳过分类 %s（csv目录无文件）：%s\n', categoryName, curCsvDir);
        continue;
    end
    fprintf('========================================\n');
    fprintf('开始处理分类 %d/%d：%s\n', folderIdx, numel(candidateCsvDirs), categoryName);
    fprintf('当前CSV目录：%s\n', curCsvDir);
    fprintf('结果保存根路径：%s\n', newSaveRoot);

    noROIFiles = {}; 

    % 循环处理每个CSV文件
    for fileIdx = 1:fileNum
        csvFile = csvFiles(fileIdx);
        csvFilePath = fullfile(csvFile.folder, csvFile.name);
        [~, fileBaseName, ~] = fileparts(csvFile.name);
        fprintf('========================================\n');
        fprintf('正在处理第 %d/%d 个文件：%s\n', fileIdx, fileNum, csvFile.name);
        
        % -------------------- 路径定义 --------------------
        curFeaImgFolder = fullfile(newSaveRoot, 'Feature_Image', fileBaseName);
        mkdirIfNotExist(curFeaImgFolder);
        curROIImgFolder = fullfile(newSaveRoot, 'ROI_Image', fileBaseName);
        mkdirIfNotExist(curROIImgFolder);
        curDownSamFolder = fullfile(newSaveRoot, 'ROI_Data', 'DownSampling', fileBaseName);
        mkdirIfNotExist(curDownSamFolder);
        curROISPFolder = fullfile(newSaveRoot, 'ROI_Data', 'ROI_SP');
        curROISFFolder = fullfile(newSaveRoot, 'ROI_Data', 'ROI_SF');
        curROISPImgFolder = fullfile(newSaveRoot, 'ROI_Data', 'ROI_SP_im');

        %% -------------------- 步骤1：读取CSV数据并转换 --------------------
        csvData = readtable(csvFilePath, 'TextType', 'string');
        s = table2array(csvData);
        [m, n] = size(s);
        fprintf('原始数据维度：%d行(时间) × %d列(距离)\n', m, n);

        %% -------------------- 步骤2：滤波、下采样、功率计算 --------------------
        new_m = floor(m / M);
        sf = zeros(new_m, n);
        sp = zeros(new_m, n);

        for index = 1:n
            temp = filter(LPF1, 1, s(:, index));
            sf(:, index) = temp(M:M:M*new_m);
            sp(:, index) = filter(LPF2, 1, sf(:, index).^2);
        end

        D = (1:n) * 5;
        T1 = (1:m) / Fs1;
        T2 = (1:new_m) / Fs2;
        fprintf('下采样后数据维度：%d行(时间) × %d列(距离)\n', new_m, n);

        %% -------------------- 步骤3：绘制特征图并保存到Feature_Image --------------------
        if FigSwitch == 1
            figure(1); set(gcf, 'outerposition', get(0, 'screensize'));
            subplot(311); imagesc(D, T1, s, [-10 10]); colorbar;
            title('waterflow of 4kHz sampling rate'); xlabel('Distance(m)'); ylabel('Time(s)');
            subplot(312); imagesc(D, T2, sf, [-10 10]); colorbar;
            title('waterflow of 500Hz sampling rate'); xlabel('Distance(m)'); ylabel('Time(s)');
            subplot(313); imagesc(D, T2, sp, [0 30]); colorbar;
            title('Power waterflow of 500Hz sampling rate'); xlabel('Distance(m)'); ylabel('Time(s)');
            print(gcf, fullfile(curFeaImgFolder, 'Downsampling'), '-djpeg', '-r300');

            figure(2); plot(T1, s(:,35), 'm'); hold on;
            plot(T1(1:M:M*new_m), sf(:,35), 'k'); hold off;
            title('Comparision between 4KHz and 500Hz at vibration event');
            xlabel('Time(s)'); ylabel('Amplitude(rad)'); axis([0 30 -30 30]); grid on;
            print(gcf, fullfile(curFeaImgFolder, 'Time Domain Waveform'), '-djpeg', '-r300');
            close;
        end

        % 保存下采样全量数据（如需开启保存，取消注释）
%         downSamplingMatPath = fullfile(curDownSamFolder, [fileBaseName '_DownSampling.mat']);
%         save(downSamplingMatPath, 'sp', 'sf', 'T2', 'T1', 'D', 's');
%         fprintf('下采样全量数据已保存至：%s\n', downSamplingMatPath);

        %% -------------------- 步骤4：ROI提取+拆分保存sp/sf+ROI图片保存 --------------------
        ROI_SP = zeros(TW*2+1, DW*2+1, 1);
        ROI_SF = zeros(TW*2+1, DW*2+1, 1);
        ROIxy = zeros(2, 1);
        RNum = 0;
        Trig = 0;
        k = 1;
        index = TW + 1;

        while(index <= new_m - TW)
            s_row = sp(index, :);

            if FigSwitch == 1
                figure(3); set(gcf, 'outerposition', get(0, 'screensize'));
                subplot(121); hold off;
                imagesc(D, T2, sp, [0 30]); colorbar;
                title('Power waterflow of 500Hz sampling rate');
                xlabel('Distance(m)'); ylabel('Time(s)');
                hold on; plot([0 max(D)],[T2(index) T2(index)],'y-.');
                
                subplot(322); hold off; bar(D, s_row, 'g');
                xlabel('Distance(m)'); ylabel('Power(rad^2)'); axis([0 max(D) 0 25]);
                hold on; plot([0 max(D)],[Th Th],'b');
            end;

            for i = DW+1 : n-DW
                winSp = sp(index-TW/2:Tstep:index+TW/2, i-DW:i+DW);
                if (s_row(i) > Th) && (s_row(i) >= max(max(winSp)))             
                    Trig = 1;
                    RNum = RNum + 1;
                    ROI_SP(:,:,RNum) = sp(index-TW:index+TW, i-DW:i+DW);
                    ROI_SF(:,:,RNum) = sf(index-TW:index+TW, i-DW:i+DW);
                    ROIxy(:,RNum) = [i, index];
                    
                    ROI_D = D(ROIxy(1,RNum)-DW:ROIxy(1,RNum)+DW);
                    ROI_T = T2(ROIxy(2,RNum)-TW:ROIxy(2,RNum)+TW);
                    roiFileName = [fileBaseName '_ROI-' num2str(RNum) '.mat'];
                    roiFigName = strrep(roiFileName, '.mat', '.jpeg');
                    
                    save(fullfile(curROISPFolder, roiFileName), 'ROI_SP', 'ROI_D', 'ROI_T', 'ROIxy');
                    save(fullfile(curROISFFolder, roiFileName), 'ROI_SF', 'ROI_D', 'ROI_T', 'ROIxy');

                    roiFigPath = fullfile(curROISPImgFolder, roiFigName);
                    fig = figure('Visible', 'off');
                    set(fig, 'Position', [100, 100, 1200, 500]);
                    pos_left = [0.05, 0.15, 0.20, 0.7];
                    pos_right = [0.35, 0.15, 0.60, 0.7];
                    axes('Position', pos_left);
                    imagesc(ROI_D, ROI_T, ROI_SP(:,:,RNum), [0 30]);
                    colorbar; axis xy; grid on;
                    title(sprintf('ROI-%d (SP Data) | Size: %dx%d', RNum, size(ROI_SP,1), size(ROI_SP,2)));
                    xlabel('Distance (m)'); ylabel('Time (s)');
                    axes('Position', pos_right);
                    imagesc(D, T2, sp, [0 30]); colorbar; axis xy; grid on;
                    title(sprintf('Global SP Data | ROI-%d Position', RNum));
                    xlabel('Distance (m)'); ylabel('Time (s)');
                    hold on;
                    roi_x1 = D(i-DW); roi_x2 = D(i+DW);
                    roi_y1 = T2(index-TW); roi_y2 = T2(index+TW);
                    rectangle('Position', [roi_x1, roi_y1, roi_x2-roi_x1, roi_y2-roi_y1], 'EdgeColor', 'red', 'LineWidth', 2);
                    plot(D(i), T2(index), 'r*', 'MarkerSize', 10, 'DisplayName', 'Peak Point');
                    legend('Location', 'best'); hold off;
                    print(fig, roiFigPath, '-djpeg', '-r300');
                    close(fig);
                    fprintf('  ├─ ROI-%d 数据/图片已保存：%s / %s\n', RNum, roiFileName, roiFigName);

                    if FigSwitch == 1    
                        subplot(121); plot([0 max(D)],[T2(index) T2(index)],'r-.');
                        subplot(322); plot(D(ROIxy(1,RNum)), s_row(ROIxy(1,RNum)),'mo');             
                        subplot(324); imagesc(ROI_D, ROI_T, ROI_SP(:,:,RNum), [0 30]); colorbar;
                        xlabel('Distance(m)'); ylabel('Time(s)');            
                        subplot(326); hold on;
                        plot(D(ROIxy(1,RNum)), T2(ROIxy(2,RNum)),'ro');
                        set(gca,'YDir','reverse');
                        xlabel('Distance(m)'); ylabel('Time(s)');
                        axis([0 max(D) 0 max(T2)]);   
                    end
                end     
            end

            drawnow;
            index = index + Tstep;

            if FigSwitch == 1 
                print(gcf, fullfile(curROIImgFolder, [fileBaseName '_ROI-' num2str(k)]),'-djpeg','-r300');   
                k = k + 1;
            end
        end
        index;
        close;

        if RNum > 0
            fprintf('当前文件共提取到 %d 个有效ROI\n', RNum);
        else
            fprintf('！！！！！！！！！当前文件未检测到满足阈值的事件，无ROI提取\n');
            noROIFiles{end+1} = csvFile.name; 
        end
    end

    fprintf('========================================\n');
    if isempty(noROIFiles)
        fprintf('✅ 分类[%s]内所有 %d 个文件均提取到有效ROI！\n', categoryName, fileNum);
    else
        fprintf('❌ 分类[%s]内共有 %d 个文件未检测到满足阈值的事件，无ROI提取，文件名如下：\n', categoryName, length(noROIFiles));
        for i = 1:length(noROIFiles)
            fprintf('  %d. %s\n', i, noROIFiles{i});
        end
    end
end

% -------------------- 脚本收尾语句 --------------------
fprintf('========================================\n');
fprintf('所有分类处理完成！\n');
fprintf('路径层级：\n');
fprintf('  ├─ Feature_Image/ (特征图，按CSV分层) \n');
fprintf('  ├─ ROI_Image/     (原ROI图片，按CSV分层) \n');
fprintf('  └─ ROI_Data/      (ROI数据根目录) \n');
fprintf('      ├─ ROI_SP/    (SP数据：直接存所有.mat文件，含原CSV名) \n');
fprintf('      ├─ ROI_SF/    (SF数据：直接存所有.mat文件，含原CSV名) \n');
fprintf('      ├─ ROI_SP_im/ (SP可视化图：直接存所有.jpeg文件，与SP数据一一对应) \n');
fprintf('      └─ DownSampling/ (下采样全量数据，按CSV分层) \n');

warning on;

%% ===================== 辅助函数：文件夹不存在则创建 =====================
function mkdirIfNotExist(folderPath)
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

%% ===================== 辅助函数：调整子图间距，避免标签重叠 =====================
function tightfig(varargin)
    fig = gcf;
    if nargin > 0
        fig = varargin{1};
    end
    set(fig, 'Position', get(fig, 'Position'));
    ax = findobj(fig, 'Type', 'Axes');
    if isempty(ax)
        return;
    end
    pos = cell2mat(get(ax, 'Position'));
    xmin = min(pos(:,1));
    ymin = min(pos(:,2));
    xmax = max(pos(:,1)+pos(:,3));
    ymax = max(pos(:,2)+pos(:,4));
    dx = xmin - 0.05;
    dy = ymin - 0.05;
    w = xmax - xmin + 0.1;
    h = ymax - ymin + 0.1;
    for i = 1:length(ax)
        p = get(ax(i), 'Position');
        p(1) = (p(1) - xmin + dx) / w;
        p(2) = (p(2) - ymin + dy) / h;
        p(3) = p(3) / w;
        p(4) = p(4) / h;
        set(ax(i), 'Position', p);
    end
    set(fig, 'Position', [fig.Position(1), fig.Position(2), fig.Position(3), fig.Position(4)]);
end

%% ===================== 辅助函数：判断分类是否已处理完成 =====================
function [skip, roiCount, origCount, roiDir] = shouldSkipCategory(categoryDir, saveCategoryDir)
    % 统计原始类别目录下的文件数（仅文件，不含子目录）
    origCount = countFiles(categoryDir);

    % ROI 数据目录候选（优先使用保存目录，其次原始目录；兼容 ROI_Data / ROI_data 两种命名）
    roiDirCandidates = {
        fullfile(saveCategoryDir, 'ROI_Data'), ...
        fullfile(saveCategoryDir, 'ROI_data'), ...
        fullfile(categoryDir, 'ROI_Data'), ...
        fullfile(categoryDir, 'ROI_data')
    };

    roiDir = '';
    for i = 1:numel(roiDirCandidates)
        if isfolder(roiDirCandidates{i})
            roiDir = roiDirCandidates{i};
            break;
        end
    end

    if isempty(roiDir)
        skip = false;
        roiCount = 0;
        return;
    end

    roiCount = countFiles(roiDir);
    skip = roiCount > origCount;
end

%% ===================== 辅助函数：统计目录下的文件数（不含子目录） =====================
function count = countFiles(folderPath)
    items = dir(folderPath);
    items = items(~[items.isdir]);
    count = numel(items);
end