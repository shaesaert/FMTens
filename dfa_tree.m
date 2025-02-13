classdef dfa_tree < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        DFA
        tree
        leafs
    end

    methods
        function G = dfa_tree(DFA,varargin)
            %dfa_tree Construct an instance of this class
            %   Arguments:
            % DFA
            G.DFA = DFA;
        end

        function initiate(G)
            %INITIATE Create tree with accepting node and its childs
            %   Arguments:
            % DFA
            [S,l]= find(G.DFA.trans==G.DFA.F);
            s=ones( length(l),1);
            t=1+cumsum(ones( length(l),1));
            q =  [G.DFA.F, S']';
            EdgeTable = table([s t],l,'VariableNames',{'EndNodes' 'l'});
            NodeTable = table(q,'VariableNames',{'q'});
            G.leafs = t'; % the new leaf nodes
            G.tree = digraph(EdgeTable, NodeTable);

        end

        function q = Lq(G,n)
            q = G.tree.Nodes.q(n);
        end

        function grow(G)
            %GROW Add children to leafs of graph

            leafs_old = G.leafs;
            for n = leafs_old
                G.growleaf(n)

            end
        end

        function [nodeIDs]=findSubtree(G,n,nodeIDs)
            

            if ~isempty(G.tree.successors(n))
                for n_next = G.tree.successors(n)'
                    [nodeIDs]=findSubtree(G,n_next,nodeIDs);
                end
            end
            nodeIDs = [nodeIDs, n];

        end
        function removeBranch(G,n)
            %REMOVEBRANCH remove branch starting from a node
            [nodeIDs]=findSubtree(G,n,[]);
            for n =nodeIDs
                G.tree = G.tree.rmnode(n);

                G.leafs = setxor( G.leafs,n);
                G.leafs = G.leafs - (G.leafs>n);
            end
           
            
            
        end


        function growleaf(G,n)
            n
            if ~sum(G.leafs == n)
                error('node is not a leaf node');
            end

            maxnode = numnodes(G.tree);
            % get q mode
            q = Lq(G,n);
            % find the possible children
            [S,l]= find(G.DFA.trans == q);

            % add nodes
            q =  S;
            NodeProps = table(S,'VariableNames',{'q'});
            G.tree = addnode(G.tree, NodeProps);

            % add edges
            s=ones( length(l),1)*n; % source node is node n
            t=maxnode+cumsum(ones( length(l),1)); % t is target node
            EdgeTable = table([s t],l,'VariableNames',{'EndNodes' 'l'});
            G.tree = addedge(G.tree, EdgeTable);

            % remove n from leaf nodes and add t
            G.leafs = G.leafs(G.leafs~=n); %remove n
            G.leafs = [G.leafs,t'];

        end
        function plot(G,varargin)
            % Plotting function for the tree structure
            if (length(varargin)>=1) && all(varargin{1} == 'letters')
                letters = G.num2label(G.DFA.act,G.tree.Edges.l);
                h = plot(G.tree, 'EdgeLabel',letters, 'NodeLabel',G.tree.Nodes.q);

            else
                h = plot(G.tree, 'EdgeLabel',G.tree.Edges.l, 'NodeLabel',G.tree.Nodes.q);
            end
        end
    end
    methods(Static)
        % convertng nummeric labels to AP based labels, not that only the
        % AP that hold are given, so !p1 is not given.
        function letters = num2label(act,Nodesl)
            letters = cell(size(Nodesl,1),1);
            for q = 1: size(Nodesl,1)
                letters{q} = act{Nodesl(q)};
            end
        end
    end
end