function [satProb_lb] = SynthesizeRobustController_decoupled_2D(sysAbs, DFA, rel, thold, tensorP,varargin)


N = 500; %maximum number of iterations performed. 
% if iscell(sysAbs)
%     sysAbs = sysAbs_total{DFA.S0};
%     rel = rel_total{DFA.S0};
% end
uhat = sysAbs.inputs;
nX = length(sysAbs.states);
nS = length(DFA.S);

% Initialise value function
% V(i,j) is the probability of reaching F from DFA state i and abstract state
% sysAbs.states(:,j)
V_lb = zeros(nS,nX);

DFA_Active = setdiff(setdiff(DFA.S,DFA.F), DFA.sink);
Converged = ones(1, nS); % create a vector with
Converged(DFA_Active)=deal(0);

V_lb(DFA.F,:) = ones(nX,1); % Set V to 1 for DFA state q = F

trans = ones(nS,nS,nX);
for k = 1:N
    % Stop iterating when all values are converged
    if min(Converged) == 1
        disp('Convergence reached!')
        disp(['Number of iteration required, k=', num2str(k)])
        break;
    end

    for i = DFA_Active(end:-1:1) % for each discrete mode
        % check if mode has converged
        if min([Converged(DFA.trans(i, :)), Converged(i)]) == 1
            continue
        else
            Converged(i)= 0;

            % Choose the correct abstract system and relation
            % if iscell(sysAbs)
            %     sysAbs = sysAbs_total{i};
            %     rel = rel_total{i};
            % end
            delta = rel.delta;
            outputs2act = rel.NonDetLabels;

            % Prepare DFA transitions for all states
            
            %for j = DFA_Active
                for l = 1: size(outputs2act,1)
                    trans(i, DFA.trans(i, l), :) = min( ...
                        shiftdim(trans(i, DFA.trans(i, l), :), 1), ...
                        1 - outputs2act(l, :));
                end
            %end
            trans = 10 * trans;

            % Choose the correct states out of V based on the DFA
            V_sort = min(max(squeeze(trans(i, :, :)), V_lb));

            % Compute value function for each action uhat
            VVlb = V_sort * sysAbs.P; % P is an object of the TensorTransitionProbability class
            % if upperBound
            %     % Compute probability of stepping to sink state for upper bound
            %     P_sink = 1 - ones(1, nX) * sysAbs.P;
            %     VVub = VVlb + P_sink;
            % end

            % Optimize over uhat and subtract delta
            V_n_lb = max(VVlb, [], 2) - delta;
            % if upperBound
            %     V_n_ub = max(VVub, [], 2) + delta;
            % end

            % Make sure that value function is between 0 and 1
            V_n_lb = min(1, max(0, V_n_lb));
            % if upperBound
            %     V_n_ub = min(1, max(0, V_n_ub));
            % end

            if min([Converged(DFA.trans(i, :))]) == 1
                Converged(i) = 1;
            elseif max(max(abs(V_n_lb' - V_lb(i, :)))) < thold
                Converged(i) = 1;
            end
            V_lb(i, :) = V_n_lb;
            % if upperBound
            %     V_ub(i, :) = V_n_ub;
            % end

        end
    end
end

satProb_lb = V_lb((0:length(V_lb)-1)*nS + DFA.trans(DFA.S0, sysAbs.labels));


end


% function [satProb_lb, pol, varargout] = SynthesizeRobustController_decoupled_2D(sysAbs, DFA, simRel, thold, tensorP,varargin)
%     % Initialize parameters and settings
%     % sysAbs.P = sysAbs.P.Prob;
% 
% 
%     % if nargin < 4 || isempty(thold)
%     %     thold = 1e-6;
%     % end
%     % 
%     % if nargin >= 5
%     %     initialonly = varargin{1};
%     % else
%     %     initialonly = false;
%     % end
%     % 
%     % if nargin >= 6
%     %     antagonist_pol = varargin{2};
%     % else
%     %     antagonist_pol = false;
%     % end
%     % 
%     % if nargin >= 7
%     %     upperBound = varargin{3};
%     % else
%     %     upperBound = false;
%     % end
%     initialonly = 1;
%     % Value iteration parameters
%     gamma = simRel.delta; % Using delta as gamma
%     maxIterations = 1000;
%     tolerance = thold;
% 
%     % Initialize value function and policy
%     nS = length(DFA.S); % Number of DFA states
%     nX = length(sysAbs.states); % Number of system abstract states
%     nu = length(sysAbs.inputs); % Number of inputs
% 
%     V_lb = zeros(nS, nX);
%     % if upperBound
%     %     V_ub = zeros(nS, nX);
%     % end
% 
%     pol = zeros(nu, nX, nS);
%     % if upperBound
%     %     pol_ub = zeros(nu, nX, nS);
%     % end
% 
%     % Initializing value function for final states
%     V_lb(DFA.F, :) = 1;
%     % if upperBound
%     %     V_ub(DFA.F, :) = 1;
%     % end
% 
%     % Value iteration process
%     for iter = 1:maxIterations
%         V_old_lb = V_lb;
%         % if upperBound
%         %     V_old_ub = V_ub;
%         % end
% 
%         for i = 1:nS
%             if ismember(i, DFA.F) || i == DFA.sink
%                 continue;
%             end
% 
%             for x = 1:nX
%                 Q_lb = zeros(nu, 1);
%                 % if upperBound
%                 %     Q_ub = zeros(nu, 1);
%                 % end
% 
%                 for u = 1:nu
%                     nextStateProbs = tensorP(:, u, x);
% 
%                     % Calculate Q-values for lower bound
%                     Q_lb(u) = sum(nextStateProbs' .* V_old_lb(i, :));
% 
%                     % if upperBound
%                     %     % Calculate Q-values for upper bound
%                     %     Q_ub(u) = sum(nextStateProbs' .* V_old_ub(i, :));
%                     % end
%                 end
% 
%                 % Update value functions
%                 V_lb(i, x) = gamma * max(Q_lb);
% 
%                 % if upperBound
%                 %     V_ub(i, x) = gamma * max(Q_ub);
%                 % end
% 
%                 % Update policies
%                 optimalControls_lb = (Q_lb == max(Q_lb));
%                 pol(:, x, i) = optimalControls_lb / sum(optimalControls_lb);
% 
%                 % if upperBound
%                 %     optimalControls_ub = (Q_ub == max(Q_ub));
%                 %     pol_ub(:, x, i) = optimalControls_ub / sum(optimalControls_ub);
%                 % end
%             end
%         end
% 
%         % Check for convergence
%         if max(max(abs(V_lb - V_old_lb))) < tolerance
%             disp(['Converged in ', num2str(iter), ' iterations']);
%             break;
%         end
% 
%         % if upperBound && max(max(abs(V_ub - V_old_ub))) < tolerance
%         if  max(max(abs(V_ub - V_old_ub))) < tolerance
%             disp(['Converged in ', num2str(iter), ' iterations (upper bound)']);
%             break;
%         end
%     end
% 
%     % Compute satisfaction probabilities
%     if initialonly
%         satProb_lb = V_lb(DFA.S0, :);
%         % if upperBound
%         %     satProb_ub = V_ub(DFA.S0, :);
%         % end
%     else
%         satProb_lb = V_lb;
%         % if upperBound
%         %     satProb_ub = V_ub;
%         % end
%     end
% 
%     % Return results
%     % varargout = {};
%     % if antagonist_pol
%     %     varargout{end+1} = pol_ub;
%     % end
%     % if upperBound
%     %     varargout{end+1} = satProb_ub;
%     % end
% 
%     disp('Finished synthesizing a robust controller');
% end
