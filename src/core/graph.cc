#include "core/graph.h"
#include "core/common.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <queue>

namespace infini
{
    void GraphObj::reconnect(const Tensor &old_input_tensor, const Tensor &new_input_tensor, 
        const Operator &op) {
        if (old_input_tensor == new_input_tensor) {
            return;
        }

        op->replaceInput(old_input_tensor, new_input_tensor); // 替换每一处位置
        old_input_tensor->removeTarget(op);
        new_input_tensor->addTarget(op); // 添加不用有重复也无所谓，删除时会删除所有重复

        // 是否还存在与原节点其他 tensor 相连接，如果没有才删除与节点的关联
        auto old_is_pred = [&] {
            for (auto &&input : op->getInputs()) {
                if (input->getSource() == old_input_tensor->getSource()) {
                    return true;
                }
            }
            return false;
        }();
        if (not old_is_pred) {
            old_input_tensor->getSource()->removeSuccessors(op);
            op->removePredecessors(old_input_tensor->getSource());
        }

        // 添加不用有重复也无所谓，删除时会删除所有重复
        if (new_input_tensor->getSource()) {
            new_input_tensor->getSource()->addSuccessors(op);
            op->addPredecessors(new_input_tensor->getSource());
        }
    }

    void GraphObj::eraseNullOpAndTensor(const Operator &op) {
        // 默认 op 输出的 Tensor 不包含 output tensor

        // 确保没有后继
        if (not op->getSuccessors().empty()) {
            return;
        }

        // 从前驱中删除后继关系
        for (auto &&input : op->getInputs()) {
            input->removeTarget(op);
        }
        for (auto &&pred : op->getPredecessors()) {
            pred->removeSuccessors(op);
        }

        // 删除节点
        for (auto &&output : op->getOutputs()) {
            removeTensor(output);
        }
        removeOperator(op);

        // 递归删除其他节点
        for (auto &&pred : op->getPredecessors()) {
            eraseNullOpAndTensor(pred);
        }
    }

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        std::set<Operator> deleted_ops;

        for (auto &op : ops) {
            if (deleted_ops.find(op) != std::end(deleted_ops)) {
                continue;
            }
            if (op->getOpType() != OpType::Transpose) {
                continue;
            }

            auto op_perm = std::static_pointer_cast<TransposeObj>(op)->getPermute();
            for (auto &&succ : op->getSuccessors()) {
                if (succ->getOpType() == OpType::Transpose) {
                    auto succ_perm = std::static_pointer_cast<TransposeObj>(succ)->getPermute();
                    if (not (succ_perm == op_perm)) {
                        continue;
                    }
                    
                    for (auto &&succ_succ : succ->getSuccessors()) {
                        reconnect(succ->getOutput(), op->getInputs()[0], succ_succ);
                    }

                    deleted_ops.emplace(op);
                    deleted_ops.emplace(succ);
                } else if (succ->getOpType() == OpType::MatMul) {
                    int op_input_rank = op->getInputs()[0]->getRank();
                    bool is_fusable = [&]() {
                        if (op_perm.back() != op_input_rank - 2 || op_perm[op_perm.size() - 2] != op_input_rank - 1) {
                            return false;
                        }
                        for (int i{}; i < op_input_rank - 2; ++i) {
                            if (i != op_perm[i]) {
                                return false;
                            }
                        }
                        return true;
                    }();
                    if (not is_fusable) {
                        continue;
                    }

                    auto matmul_succ = std::static_pointer_cast<MatmulObj>(succ);
                    if (op == succ->getInputs()[1]->getSource()) {
                        matmul_succ->setTransB(not matmul_succ->getTransB());
                    }
                    
                    if (op == succ->getInputs()[0]->getSource()) {
                        matmul_succ->setTransA(not matmul_succ->getTransA());
                    }
                    reconnect(op->getOutput(), op->getInputs()[0], succ);

                    deleted_ops.emplace(op);
                }
            }
        }
        for (auto &&op : deleted_ops) {
            eraseNullOpAndTensor(op);
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        std::vector<size_t> offsets;
        offsets.reserve(offsets.size());

        for (auto &tensor : tensors) {
            offsets.emplace_back(allocator.alloc(tensor->getBytes()));
        }
        void *mem = allocator.getPtr();
        for (size_t i{}; i < tensors.size(); ++i) {
            auto &tensor = tensors[i];
            Blob blob = std::make_shared<BlobObj>(runtime, static_cast<std::byte *>(mem) + offsets[i]);
            tensor->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini