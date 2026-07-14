% ====================== DPALA_Fixed (Dual-Population ALA + Fixed LSHADE) ======================
% 双种群算法：子种群1使用原ALA，子种群2使用LSHADE（禁用种群规模自适应缩减）
% 固定两个子种群规模不变，彻底避免因规模变化导致的索引/维度/Position异常问题
% 兼容原7个输出参数，确保Position始终为数值行向量
% 论文标题可写：HALA: A Hybrid Dual-Population Optimizer Based on ALA and LSHADE
function [Score, Position, Conv_curve, History_pos, Avg_curve, Div_curve, Traj_dim1] = ...
    HALA_optimizer(N, Max_iter, lb, ub, dim, fobj)

    % 参数设置
    num_subpop = 2;                          % 双种群
    sub_N = floor(N / num_subpop);            % 每个子种群规模（总个体数 ≈ N）
    if mod(N, num_subpop) == 1
        sub_N = sub_N + [1, 0];               % 微调让总和等于N（可选）
    end
    
    migration_interval = 20;                 % 迁移间隔（可调10~30）
    %migration_rate = 2;                      % 每次迁移交换的精英数（1~3）

    % 初始化子种群
    X = cell(1, num_subpop);
    fitness = cell(1, num_subpop);
    Position = cell(1, num_subpop);
    Score = inf(1, num_subpop);
    
    % 子种群1: ALA
    X{1} = initialization(sub_N, dim, ub, lb);
    fitness{1} = zeros(sub_N, 1);
    Position{1} = zeros(1, dim);
    Score(1) = inf;
    for i = 1:sub_N
        fitness{1}(i) = fobj(X{1}(i,:));
        if fitness{1}(i) < Score(1)
            Position{1} = X{1}(i,:);
            Score(1) = fitness{1}(i);
        end
    end
    
    % 子种群2: LSHADE（固定规模）
    X{2} = initialization(sub_N, dim, ub, lb);
    fitness{2} = zeros(sub_N, 1);
    Position{2} = zeros(1, dim);
    Score(2) = inf;
    for i = 1:sub_N
        fitness{2}(i) = fobj(X{2}(i,:));
        if fitness{2}(i) < Score(2)
            Position{2} = X{2}(i,:);
            Score(2) = fitness{2}(i);
        end
    end
    
    % LSHADE参数（固定规模，不缩减）
    H = 6;                                   % 记忆大小
    memory_F = 0.5 * ones(H, 1);
    memory_CR = 0.5 * ones(H, 1);
    archive = [];                            % 外部存档
    arc_rate = 1.4;
    arc_size = round(arc_rate * sub_N);
    p_best_rate = 0.11;                      % pbest比例
    
    % 全局最佳
    [global_best_score, best_sub] = min(Score);
    global_best_pos = Position{best_sub}(:)';
    
    % 输出变量
    Conv_curve = zeros(1, Max_iter);
    History_pos = zeros(N, dim, Max_iter);
    Avg_curve = zeros(1, Max_iter);
    Div_curve = zeros(1, Max_iter);
    Traj_dim1 = zeros(1, Max_iter);
    
    vec_flag = [1, -1];
    
    for Iter = 1:Max_iter
        %% 子种群1: 原版ALA更新
        RB = randn(sub_N, dim);
        F_flag = vec_flag(randi(2));
        theta = 2 * atan(1 - Iter/Max_iter);
        
        Xnew = zeros(sub_N, dim);
        for i = 1:sub_N
            E = 2 * log(1/rand) * theta;
            if E > 1
                if rand < 0.3
                    r1 = 2*rand(1,dim) - 1;
                    r = randi(sub_N);
                    Xnew(i,:) = Position{1} + F_flag .* RB(i,:) .* (r1.*(Position{1}-X{1}(i,:)) + (1-r1).*(X{1}(i,:)-X{1}(r,:)));
                else
                    r2 = rand() * (1 + sin(0.5*Iter));
                    r = randi(sub_N);
                    Xnew(i,:) = X{1}(i,:) + F_flag .* r2 * (Position{1} - X{1}(r,:));
                end
            else
                if rand < 0.5
                    radius = sqrt(sum((Position{1} - X{1}(i,:)).^2));
                    r3 = rand();
                    spiral = radius * (sin(2*pi*r3) + cos(2*pi*r3));
                    Xnew(i,:) = Position{1} + F_flag .* X{1}(i,:) .* spiral * rand;
                else
%                     G = 2 * (sign(rand-0.5)) * (1 - Iter/Max_iter);
%                     Xnew(i,:) = Position{1} + F_flag .* G .* Levy(dim) .* (Position{1} - X{1}(i,:));
                    
                    % 改进：引入平方衰减 + t 分布缩放（EALA 风格）
%                     dof = 10 * (1 - Iter/Max_iter) + 1;  % 自由度从10衰减到1（重尾→轻尾）
%                     t_scale = trnd(dof, 1, dim);         % t 分布随机数
%                     adapt_G = 2 * (1 - Iter/Max_iter)^2; % 平方衰减，更慢减小步长
%                     G = adapt_G * sign(t_scale);         % 用 t 分布方向代替随机 sign
%                     Xnew(i,:) = Position{1} + F_flag .* G .* Levy(dim) .* (Position{1} - X{1}(i,:));
                    
                    % === 方式3：t 分布 + Levy 混合扰动（EALA 核心）===
                    dof = 10 * (1 - Iter/Max_iter) + 1;               % 自由度衰减
                    t_step = trnd(dof, 1, dim);                        % t 分布步长
                    levy_step = Levy(dim);                             % Levy 步长
                    combined_step = 0.7 * t_step + 0.3 * levy_step;    % 混合（0.7 t + 0.3 Levy）
                    adapt_factor = 1.5 * (1 - Iter/Max_iter);          % 自适应因子（线性衰减）
                    Xnew(i,:) = Position{1} + F_flag .* adapt_factor .* combined_step .* (Position{1} - X{1}(i,:));
                end
            end
        end
        
        for i = 1:sub_N
            Xnew(i,:) = max(min(Xnew(i,:), ub), lb);
            newfit = fobj(Xnew(i,:));
            if newfit < fitness{1}(i)
                X{1}(i,:) = Xnew(i,:);
                fitness{1}(i) = newfit;
            end
            if fitness{1}(i) < Score(1)
                Position{1} = X{1}(i,:);
                Score(1) = fitness{1}(i);
            end
        end
        
        %% 子种群2: LSHADE更新（固定规模，无缩减）
        mem_idx = randi(H, sub_N, 1);
        F_vec = memory_F(mem_idx) + 0.1 * randn(sub_N, 1);
        F_vec = max(min(F_vec, 1), 0);
        CR_vec = memory_CR(mem_idx) + 0.1 * randn(sub_N, 1);
        CR_vec = max(min(CR_vec, 1), 0);
        
        [~, sort_idx] = sort(fitness{2});
        success_F = [];
        success_CR = [];
        
        Xnew = zeros(sub_N, dim);
        for i = 1:sub_N
            np = max(round(p_best_rate * sub_N), 2);
            pbest_idx = sort_idx(randi(np));
            
            r1 = randi(sub_N);
            while r1 == i
                r1 = randi(sub_N);
            end
            
            if ~isempty(archive)
                r2 = randi(size(archive,1));
                xr = archive(r2,:);
            else
                r2 = randi(sub_N);
                while r2 == i || r2 == r1
                    r2 = randi(sub_N);
                end
                xr = X{2}(r2,:);
            end
            
            v = X{2}(i,:) + F_vec(i) * (X{2}(pbest_idx,:) - X{2}(i,:)) + F_vec(i) * (xr - X{2}(r1,:));
            
            jrand = randi(dim);
            u = X{2}(i,:);
            for j = 1:dim
                if rand <= CR_vec(i) || j == jrand
                    u(j) = v(j);
                end
            end
            
            u = max(min(u, ub), lb);
            newfit = fobj(u);
            
            if newfit <= fitness{2}(i)
                archive = [archive; X{2}(i,:)];
                X{2}(i,:) = u;
                fitness{2}(i) = newfit;
                success_F = [success_F; F_vec(i)];
                success_CR = [success_CR; CR_vec(i)];
            end
            
            if fitness{2}(i) < Score(2)
                Position{2} = X{2}(i,:);
                Score(2) = fitness{2}(i);
            end
        end
        
        % 更新存档和历史记忆
        if size(archive,1) > arc_size
            archive = archive(randperm(size(archive,1), arc_size), :);
        end
        if ~isempty(success_F)
            k = mod(Iter, H) + 1;
            memory_F(k) = mean(success_F);
            memory_CR(k) = mean(success_CR);
        end
        
        %% 迁移操作（每migration_interval代交换精英）
        if mod(Iter, migration_interval) == 0
            % 找两个子种群的最差个体
            [~, worst1] = max(fitness{1});
            [~, worst2] = max(fitness{2});
            
            % 双向迁移：互相用对方的最佳替换最差
            X{1}(worst1,:) = Position{2};
            fitness{1}(worst1) = fobj(X{1}(worst1,:));
            
            X{2}(worst2,:) = Position{1};
            fitness{2}(worst2) = fobj(X{2}(worst2,:));
        end
        
        %% 更新全局最佳
        [global_best_score, best_sub] = min(Score);
        global_best_pos = Position{best_sub}(:)';
        
        %% 记录曲线
        all_X = [X{1}; X{2}];
        all_fitness = [fitness{1}; fitness{2}];
        
        pad = N - size(all_X,1);
        if pad > 0
            all_X = [all_X; zeros(pad, dim)];
            all_fitness = [all_fitness; inf(pad,1)];
        end
        
        History_pos(:,:,Iter) = all_X;
        Conv_curve(Iter) = global_best_score;
        Avg_curve(Iter) = mean(all_fitness(:));
        Div_curve(Iter) = mean(sqrt(sum((all_X - global_best_pos).^2, 2)));
        Traj_dim1(Iter) = global_best_pos(1);
    end
    
    Score = global_best_score;
    Position = global_best_pos;  % 最终为1×dim数值行向量
end

% Levy飞行函数
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1,d)*sigma;
    v = randn(1,d);
    step = u ./ abs(v).^(1/beta);
    o = step;
end