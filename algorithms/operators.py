"""Các toán tử không sửa đầu vào; hàm có dấu _ nhận dữ liệu đã kiểm tra."""
import numpy as np
from numpy.random import Generator
from models import Permutation
from utils.validation import integer, permutation, population_data

# Các hàm có dấu _ thực hiện toán tử, dùng trong vòng lặp GA.
# Các hàm công khai kiểm tra dữ liệu trước khi gọi phần thực hiện tương ứng.

# CHỌN LỌC CHA/MẸ BẰNG TOURNAMENT.
def _tournament(population, distances, size, rng):
    # Lấy ngẫu nhiên size cá thể khác nhau trong một nhóm thi đấu.
    candidates = rng.choice(len(population), size=size, replace=False).tolist()
    # Thắng = quãng đường ngắn nhất; nếu bằng nhau, ưu tiên chỉ số nhỏ hơn.
    winner = min(candidates, key=lambda i: (distances[i], i))
    return population[winner].copy(), candidates, winner

def tournament_selection(population, distances, size: int, rng: Generator) -> Permutation:
    routes, costs = population_data(population, distances)
    integer(size, "Tournament", 2)
    if size > len(routes):
        raise ValueError("Tournament không được lớn hơn quần thể.")
    return _tournament(routes, costs, size, rng)[0]

# LAI GHÉP OX (ORDER CROSSOVER): kết hợp thứ tự điểm của hai cha mẹ.
def _ox(a, b, rng):
    n = len(a)
    # Chọn hai biên: đoạn giữ lại là [left:right], không gồm vị trí right.
    left, right = sorted(rng.choice(n + 1, size=2, replace=False).tolist())
    def child(first, second):
        # Bước 1: giữ nguyên một đoạn của cha/mẹ thứ nhất.
        # Số 0 chỉ là chỗ trống tạm thời, sẽ được điền hết trước khi trả con.
        result = [0] * n
        result[left:right] = first[left:right]
        used = set(first[left:right])
        # Bước 2: duyệt vòng từ biên phải theo thứ tự cha/mẹ thứ hai.
        order = [(right + offset) % n for offset in range(n)]
        positions = [i for i in order if not left <= i < right]
        # Bỏ các mã đã có trong đoạn giữ lại để không tạo điểm trùng.
        genes = [second[i] for i in order if second[i] not in used]
        # Bước 3: điền các mã còn lại vào vị trí trống theo thứ tự duyệt.
        for position, gene in zip(positions, genes):
            result[position] = gene
        return result
    # Đổi vai trò hai cha mẹ để tạo con thứ hai.
    return (child(a, b), child(b, a)), (left, right)

def ordered_crossover(parent_a: Permutation, parent_b: Permutation, rng: Generator):
    a = permutation(parent_a)
    b = permutation(parent_b, len(a))
    return _ox(a, b, rng)[0]

# ĐỘT BIẾN SWAP: đổi chỗ hai điểm, không thêm hoặc xóa điểm giao hàng.
# Xác suất đột biến được kiểm tra ở run_ga(), trước khi gọi hàm này.
def _swap(route, rng):
    result = route.copy()
    i, j = rng.choice(len(result), size=2, replace=False).tolist()
    result[i], result[j] = result[j], result[i]
    return result, (i, j)

def swap_mutation(route: Permutation, rng: Generator) -> Permutation:
    return _swap(permutation(route), rng)[0]

# GIỮ CÁ THỂ TỐT (ELITISM): hàm riêng để sử dụng/kiểm thử toán tử.
# Trong run_ga(), thao tác tương đương được thực hiện trực tiếp trên quần thể.
def select_elites(population, distances, elite_size: int):
    routes, costs = population_data(population, distances)
    integer(elite_size, "Số elite", 1)
    if elite_size >= len(routes):
        raise ValueError("Số elite phải nhỏ hơn quần thể.")
    return [routes[i].copy() for i in np.argsort(costs, kind="stable")[:elite_size]]
