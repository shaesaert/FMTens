%%% Use case: Highway Overtake 
%
% Visualize the DFA and polytopes in this use case

clc
clear
close all

warning('off','all')

config

%% Specify system parameters and regions

% Set up an LTI model
sysLTI = trafficSys(1);
disp("LTI model setup")

% Bounds on state space [x, y, v_x, v_y]
state_lb = [-20,-1.8,-11,-1];
state_ub = [20,5.4,11,1];
sysLTI.X = Polyhedron('lb', state_lb', 'ub', state_ub');

% Define bounds on input space
sysLTI.U = Polyhedron('lb', [-2 -0.5], 'ub', [1.5 0.5]);
disp("Input and state constraints")

%% Translate the specification

% Setup the specification
[formula, sysLTI] = overtakeSpec(sysLTI);
disp("Specification created")

% Translate the spec to a DFA
[DFA] = TranslateSpec(formula,sysLTI.AP);

%% Visualize the DFA

% Buchi Automaton
dfaVisualization(DFA);

% Define polytopes of satisfying and not satisfying specifications
spec.p1 = Polyhedron('lb',[-20,-1.8],'ub',[20,1.8]);
spec.p2 = Polyhedron('lb',[5,-1.8],'ub',[20,5.4]);
spec.p3 = Polyhedron('lb',[-20,-1.8],'ub',[-5,5.4]);
spec.np1 = Polyhedron('lb',[-20,1.8],'ub',[20,5.4]);
spec.np2 = Polyhedron('lb',[-20,-1.8],'ub',[5,5.4]);
spec.np3 = Polyhedron('lb',[-5,-1.8],'ub',[20,5.4]);

% Polytopes of specification
figure;
spec.p1.plot('alpha',0.3,'color','blue'); hold on;
spec.p2.plot('alpha',0.3,'color','green'); hold on;
spec.p3.plot('alpha',0.3,'color','red'); hold off;
legend('P1','P2','P3');xlabel("X direction");ylabel("Y direction");title("Full Polytope Visualization");

% Polytopes staying at certain state
figure;
colors = {'red','blue','green','yellow','orange','purple'};
actset = DFA.act(DFA.trans(DFA.S0,:)==2); % Set of actions to stay in initial state
for i = 1:length(DFA.act)
    action = DFA.act{i};
    transition = DFA.trans(DFA.S0,i);
    if transition == DFA.S0
        % Check what specification is inside action
        p1Bool = contains(action,'p1');
        p2Bool = contains(action,'p2');
        p3Bool = contains(action,'p3');
        % Define which polyhedron to use
        if p1Bool
            p1Ind = 'p1';
        else
            p1Ind = 'np1';
        end
        if p2Bool
            p2Ind = 'p2';
        else
            p2Ind = 'np2';
        end
        if p3Bool
            p3Ind = 'p3';
        else
            p3Ind = 'np3';
        end

        % Create polytope to satisfy specification
        poly = intersect(spec.(p1Ind),intersect(spec.(p2Ind),spec.(p3Ind)));
        
        % If polytope is empty remove name from action set
        if ~poly.isEmptySet()
            poly.plot('alpha',0.3,'color',colors{i}); hold on;
        else
            actset(i) = [];
        end
    else
        continue
    end
end
legend(actset);xlabel("X position");ylabel("Y position");title("Polytopes of staying at state");hold off;

% Visualize polytopes
combs = [1 0 1;
         0 0 1;
         0 0 0;
         0 1 0;
         1 0 0;
         1 1 0];
figure;
k=1;
for i = 1:size(combs,1)
    comb = combs(i,:);
    name = "";
    for j = 1:length(comb)
        if comb(j) == 1
            polytope(j) = spec.("p"+num2str(j));
        else
            polytope(j) = spec.("np"+num2str(j));
        end
    end
    int = intersect(polytope(3), intersect(polytope(1),polytope(2)));
    if ~int.isEmptySet()
        int.plot('alpha',0.3,'color',colors{k}); hold on;
        k=k+1;
    end
end
% title("Polytopes defined by letters (within state space)");
xlabel("Relative x position");ylabel("Relative y position")
legend({'$$ l_1 $$','$$ l_2 $$','$$ l_3 $$','$$ l_4 $$','$$ l_6 $$','$$ l_7 $$'},'Interpreter','latex');