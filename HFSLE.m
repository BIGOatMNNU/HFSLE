function [feature_slct,W,YD] = HFSLE(X, Y, Z ,tree, lambda, alpha, theta, beta, flag)
% rand('seed',1);
% internalNodes(find(internalNodes==-1))=[];
indexRoot = tree_Root(tree);% The root of the tree
internalNodes = tree_InternalNodes(tree);
leafNodes  = tree_LeafNode( tree );
eps = 1e-8; % set your own tolerance
maxIte = 10;
noi_nl = [];
th=0.5;
[~,d] = size(X{indexRoot}); % get the number of features
for noi = 1:length(internalNodes)
    if isempty(Y{internalNodes(noi)})
        noi_nl = [noi_nl, noi];
        W{internalNodes(noi)} = rand(d, length(get_children_set(tree, noi)));
    end
end
nosamplenode=internalNodes(noi_nl);
internalNodes(noi_nl) = [];
noLeafNode =[internalNodes;indexRoot];
for i = 1:length(noLeafNode)
    m(noLeafNode(i)) = length(get_children_set(tree, noLeafNode(i)));
end
% maxm=max(m);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=1;

%% initialize
for j = 1:length(noLeafNode)
    
    Y{noLeafNode(j)}=conversionY01_extendlhy(Y{noLeafNode(j)},m(noLeafNode(j)),noLeafNode(j),tree);%extend 2 to [1 0]
    W{noLeafNode(j)} = rand(d, m(noLeafNode(j))); % initialize W
    %%
    XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
    child=get_children_set(tree, noLeafNode(j));
    gapsum=0;
    for k=1:length(child)
        cur_node=child(k);
        sib=setdiff(child,cur_node);
        cur_node_des=tree_Descendant( tree,cur_node );
        cur_node_leaf=cur_node_des(ismember(cur_node_des,leafNodes));
        for q=1:length(cur_node_leaf)
            if isempty(X{cur_node_leaf(q)})
                continue
            end
            gapsum=gapsum+mean(X{cur_node_leaf(q)},1)-mean(X{cur_node},1);
        end
    end
    gap{noLeafNode(j)}=gapsum;
    gTg{noLeafNode(j)}=gapsum'*gapsum;
    [cn,cm]=size(Y{noLeafNode(j)});
    YD{noLeafNode(j)} = Y{noLeafNode(j)};
    I{noLeafNode(j)}=eye(cn,cn);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:10
    %% Update the root node
    %% initialization
    for j = 1:length(noLeafNode)
        D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));
    end
    YD_old=YD;
    %% Update the root node
    
    W{indexRoot} = inv(XX{indexRoot} + lambda * D{indexRoot}+theta*gTg{indexRoot}) * (X{indexRoot}'*YD{indexRoot});
    YD{indexRoot}=inv((1+alpha)*I{indexRoot})*(X{indexRoot}*W{indexRoot}+alpha*Y{indexRoot});
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update the internal nodes
    
    for j = 1:length(internalNodes)
        if ~isempty(W{internalNodes(j)})
            W{internalNodes(j)} = inv(XX{internalNodes(j)} + lambda * D{internalNodes(j)}+theta*gTg{internalNodes(j)}) * (X{internalNodes(j)}'*YD{internalNodes(j)});
            Lsum=0;
            PYD=tree(internalNodes(j),1);
            indx=Z{PYD}(ismember(Z{PYD},Z{internalNodes(j)}));
            CYD=YD{PYD}(indx,:);
            C1 = corr(CYD',CYD','type','pearson');
            C=zeros(length(indx),length(indx));
            for k=1:size(C1,1)
                neg=find(C1(k,:)<0);
                C1(k,neg)=0;
                [~,des]=sort(C1(k,:),'descend');
                desnum=length(des);
                if desnum>=10
                    Kneig=10;
                else
                    Kneig=desnum;
                end
                des(find(des==k))=[];
                C(k,des(1:Kneig))=C1(k,des(1:Kneig));
                C(des(1:Kneig),k)=C1(k,des(1:Kneig))';
            end
            D1 = diag(sum(C,2));
            L=D1-C;
            Lsum=Lsum+L;
            YD{internalNodes(j)}=inv((1+alpha)*I{internalNodes(j)}+beta*Lsum)*(X{internalNodes(j)}*W{internalNodes(j)}+alpha*Y{internalNodes(j)});
            cur_L{internalNodes(j)}=Lsum;
        end
    end
    %% Print the value of object function if flag is 1.
    if (flag ==1)
        obj(i)=norm(X{indexRoot}*W{indexRoot}-YD{indexRoot},'fro')^2+lambda*L21(W{indexRoot})+alpha*norm(YD{indexRoot}-Y{indexRoot},'fro')^2+ theta*norm(gap{indexRoot}*W{indexRoot})^2;
        for j = 1:length(internalNodes)
            obj(i)=obj(i)+norm(X{internalNodes(j)}*W{internalNodes(j)}-YD{internalNodes(j)},'fro')^2+lambda*L21(W{internalNodes(j)})+alpha*norm(YD{internalNodes(j)}-Y{internalNodes(j)},'fro')^2+beta*trace(YD{internalNodes(j)}'*cur_L{internalNodes(j)}*YD{internalNodes(j)})+ theta*norm(gap{internalNodes(j)}*W{internalNodes(j)})^2;
        end
    end
end

noLeafNode=[noLeafNode;nosamplenode];
for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:end);
end
if (flag == 1)
    fontsize = 20;
    figure1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',figure1,'FontSize',fontsize,'FontName','Times New Roman');
    
    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    xlim(axes1,[0.8 10]);
    %     ylim(axes1,[16000,36000]);%Cifar
    % set(gca,'yscale','log')
    set(gca,'FontName','Times New Roman','FontSize',fontsize);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end



