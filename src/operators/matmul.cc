#include "operators/matmul.h"
#include "core/common.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Shape A = inputs[0]->getDims();
        int a { transA ? A.back() : A[A.size() - 2]};
        Shape B = inputs[1]->getDims();
        int b = { transB ? B[B.size() - 2] : B.back() };

        A.resize(A.size() - 2);
        B.resize(B.size() - 2);
        Shape res = infer_broadcast(A, B);

        res.emplace_back(a);
        res.emplace_back(b);

        return {{res}};
    }

} // namespace infini