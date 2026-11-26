"""2-opt first improvement với kho cố định và số lượt hữu hạn."""
from models import DistanceMatrix, Permutation
from utils.validation import matrix, permutation, two_opt_config
from utils.distance import _route_distance

# PHẦN MỞ RỘNG: tìm kiếm cục bộ 2-opt trên một tuyến GA đã tìm được.
# Chỉ nhận thay đổi làm giảm quãng đường; nếu không có thì giữ nguyên tuyến.
def _two_opt(route, d, max_passes, epsilon):
    # Sao chép để không sửa trực tiếp tuyến đầu vào.
    best = route.copy()
    cost = _route_distance(best, d)
    for _ in range(max_passes):
        # Khép kín hành trình để xét cả các cạnh nối với kho.
        closed = [0, *best, 0]
        improved = False
        for i in range(1, len(closed) - 2):
            for j in range(i + 1, len(closed) - 1):
                # Thử thay hai cạnh (a,b), (c,e) bằng (a,c), (b,e).
                # Đảo đoạn từ b đến c để vẫn tạo một chu trình qua đủ điểm.
                a, b, c, e = closed[i - 1], closed[i], closed[j], closed[j + 1]
                # delta = độ dài mới - độ dài cũ của hai cạnh.
                # Ma trận đối xứng nên đảo các cạnh bên trong không đổi độ dài.
                delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                # epsilon là ngưỡng nhỏ để tránh nhận sai do sai số số thực.
                if delta < -epsilon:
                    candidate = best.copy()
                    candidate[i - 1:j] = reversed(candidate[i - 1:j])
                    # Tính lại toàn tuyến để xác nhận thực sự ngắn hơn.
                    new_cost = _route_distance(candidate, d)
                    if new_cost < cost - epsilon:
                        # FIRST IMPROVEMENT: nhận cách cải thiện đầu tiên tìm thấy,
                        # kết thúc lượt hiện tại rồi bắt đầu lượt tìm kiếm mới.
                        best, cost, improved = candidate, new_cost, True
                        break
            if improved:
                break
        # Không có phép đổi tốt hơn trong cả lượt: dừng sớm.
        # Không đồng nghĩa tuyến đã tối ưu toàn cục.
        if not improved:
            break
    return best

def two_opt(route: Permutation, distance_matrix: DistanceMatrix,
            max_passes: int = 20, improvement_epsilon: float = 1e-9) -> Permutation:
    d = matrix(distance_matrix)
    route = permutation(route, len(d) - 1)
    two_opt_config(max_passes, improvement_epsilon)
    return _two_opt(route, d, max_passes, improvement_epsilon)
