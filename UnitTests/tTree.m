%% Test Class Definition
classdef tTree < matlab.unittest.TestCase
%% unfinished test case 
    properties
        DFA
        
    end

    methods  (TestClassSetup)
        function createArguments(testCase)
            %% Synthesize scLTL formula (or input DFA yourself)
            %%% use LTL2BA and check if determinstic and accepting state with loop with 1.
            % input: (sc)LTL formula and atomic propositions (see readme in folder
            % LTL2BA)
            % output: struct DFA containing (among other) the transitions
            AP = {'p1', 'p2', 'p3'};
            formula = '( (!p2 | !p3 ) U p1)';  % p1 = parking, p2 = avoid region
            % formula should use atomic propositions in sysLTI.AP. 
            
            % Make sure your current folder is the main SySCoRe folder
            [testCase.DFA] = TranslateSpec(formula,AP);
            
          
           
        end
    end
    %% Test Method Block
    methods (Test)
        %% Test basic functinality of EReachTime
        function testBasicLoad(testCase)
            createArguments(testCase)

            testCase.DFA
 
           
        end
        function testTree(testCase)
            createArguments(testCase)

            G = dfa_tree(testCase.DFA);
            G.initiate()
 
        end
       function testGrow(testCase)
            createArguments(testCase)

            G = dfa_tree(testCase.DFA);
            G.initiate()
            G.grow()
        
                        
       end 

       function testGrowleaf(testCase)
            createArguments(testCase)

            G = dfa_tree(testCase.DFA);
            G.initiate()
            G.growleaf(2)
        
                        
        end
        function testPlot(testCase)
            createArguments(testCase)

            G = dfa_tree(testCase.DFA);
            G.initiate()
            plot(G)
            

        end
         function testPlotletter(testCase)
            createArguments(testCase)

            G = dfa_tree(testCase.DFA);
            G.initiate()
           
             plot(G,'letters')

         end



         function testremoveBranch(testCase)
             createArguments(testCase)
             G = dfa_tree(testCase.DFA);
             G.initiate()
             removeBranch(G,2)
             G.grow()
             plot(G,'letters')

         end

         function testFindSubtree(testCase)
             createArguments(testCase)
             G = dfa_tree(testCase.DFA);
             G.initiate()
             [nodeIDs]=findSubtree(G,2,[])
         end

    end
end