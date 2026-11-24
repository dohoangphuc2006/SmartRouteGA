"""Điều phối GA/Hybrid, lưu hội tụ và ví dụ thao tác có giới hạn."""
from collections import Counter
from time import perf_counter
import numpy as np
from numpy.random import Generator
from models import Algorithm, DistanceMatrix, GAConfig, GenerationRecord, OperatorTrace, OptimizationResult
from algorithms.operators import _ox, _swap, _tournament
from algorithms.two_opt import _two_opt
from utils.distance import _population_distances, _route_distance
from utils.validation import ga_config, integer, matrix

# CÁCH ĐỌC: run_ga() là luồng chính; các toán tử nằm trong operators.py,
# phần mở rộng 2-opt nằm trong two_opt.py; hàm đánh giá nằm trong utils/distance.py.
# Một cá thể là thứ tự giao hàng, ví dụ [3, 1, 2], chưa chứa kho 0.

# KHỞI TẠO QUẦN THỂ: tạo nhiều hoán vị ngẫu nhiên của các mã điểm 1..n.
# Mỗi cá thể chứa đủ điểm, không lặp và không thiếu điểm giao hàng.
def initialize_population(n_points: int, population_size: int, rng: Generator):
    integer(n_points, "Số điểm", 2)
    integer(population_size, "Số cá thể", 2)
    return [rng.permutation(np.arange(1, n_points + 1)).tolist() for _ in range(population_size)]

def run_ga(distance_matrix: DistanceMatrix, config: GAConfig, algorithm: Algorithm = "ga") -> OptimizationResult:
    # BƯỚC 1 - KIỂM TRA ĐẦU VÀO: ma trận khoảng cách và cấu hình GA hợp lệ.
    d = matrix(distance_matrix)
    ga_config(config, algorithm)
    started = perf_counter()
    # BƯỚC 2 - KHỞI TẠO: seed giúp tái lập các lựa chọn ngẫu nhiên.
    rng = np.random.default_rng(config.seed)
    population = initialize_population(len(d) - 1, config.population_size, rng)
    # BƯỚC 3 - ĐÁNH GIÁ: tính quãng đường khép kín của từng cá thể.
    # Tuyến ngắn hơn là tốt hơn. Dùng trực tiếp chi phí để tối thiểu hóa,
    # thay vì đổi dấu thành fitness và tối đa hóa như ví dụ trong bài lab.
    costs = _population_distances(population, d)
    index = int(np.argmin(costs))
    best, best_cost = population[index].copy(), float(costs[index])
    history = [GenerationRecord(0, best_cost)]
    trace, counts = [], Counter()
    # GHI VÍ DỤ cho tab Các bước thuật toán; không tham gia chọn nghiệm.
    def record(generation, operator, before, after, details):
        if counts[operator] < config.trace_limit_per_operator:
            trace.append(OperatorTrace(generation, operator, [r.copy() for r in before],
                                       [r.copy() for r in after], details))
            counts[operator] += 1
    # BƯỚC 4 - TIẾN HÓA: lặp qua số thế hệ đã cấu hình.
    for generation in range(1, config.generations + 1):
        # GIỮ CÁ THỂ TỐT (ELITISM): sao chép các tuyến ngắn nhất sang thế hệ mới.
        # Các bản sao elite không đi qua lai ghép/đột biến trong thế hệ này.
        elite_indices = np.argsort(costs, kind="stable")[:config.elite_size]
        next_population = [population[i].copy() for i in elite_indices]
        while len(next_population) < config.population_size:
            # CHỌN LỌC CHA MẸ (TOURNAMENT): tổ chức hai lần chọn độc lập.
            # Mỗi lần lấy ngẫu nhiên k cá thể, chọn tuyến ngắn nhất trong nhóm.
            parents = []
            for _ in range(2):
                parent, candidates, winner = _tournament(population, costs, config.tournament_size, rng)
                parents.append(parent)
                record(generation, "selection", [population[i] for i in candidates], [parent],
                       {"candidate_indices": candidates, "distances": [float(costs[i]) for i in candidates],
                        "winner_index": winner})
            # LAI GHÉP (OX): với xác suất pc, kết hợp hai cha mẹ thành hai con.
            # OX giữ các con là hoán vị hợp lệ, không lặp/thiếu điểm giao hàng.
            if rng.random() < config.crossover_rate:
                children, cuts = _ox(*parents, rng)
                record(generation, "ox", parents, list(children), {"cut_left": cuts[0], "cut_right_exclusive": cuts[1]})
            else:
                # Không lai ghép: tạo con bằng bản sao cha mẹ.
                children = tuple(parent.copy() for parent in parents)
            for child in children:
                # Không thêm dư con khi quần thể đã đạt kích thước yêu cầu.
                if len(next_population) == config.population_size:
                    break
                # ĐỘT BIẾN (SWAP): với xác suất pm, đổi chỗ hai điểm của con.
                # Tạo biến thể mới nhưng vẫn giữ mỗi điểm đúng một lần.
                if rng.random() < config.mutation_rate:
                    mutated, positions = _swap(child, rng)
                    record(generation, "mutation", [child], [mutated], {"positions": list(positions)})
                    child = mutated
                next_population.append(child)
        # THAY THẾ THẾ HỆ VÀ ĐÁNH GIÁ LẠI toàn bộ quần thể mới.
        population = next_population
        costs = _population_distances(population, d)
        index = int(np.argmin(costs))
        # PHẦN MỞ RỘNG - HYBRID GA + 2-OPT (GA cơ bản bỏ qua khối này).
        # Định kỳ K thế hệ hoặc ở thế hệ cuối, cải thiện cục bộ một tuyến
        # tốt nhất hiện tại bằng 2-opt; tối đa M lượt tìm cải thiện.
        # 2-opt không làm dài tuyến đầu vào, nhưng thay đổi quần thể về sau:
        # không bảo đảm Hybrid luôn thắng một lần chạy GA cơ bản riêng biệt.
        if algorithm == "hybrid_ga" and (generation % config.two_opt_interval == 0 or generation == config.generations):
            before = population[index]
            after = _two_opt(before, d, config.two_opt_max_passes, config.improvement_epsilon)
            new_cost = _route_distance(after, d)
            record(generation, "two_opt", [before], [after],
                   {"distance_before": float(costs[index]), "distance_after": new_cost})
            # Thay tuyến trong quần thể để thế hệ sau kế thừa kết quả 2-opt.
            population[index], costs[index] = after, new_cost
        # LƯU NGHIỆM TỐT NHẤT từ đầu đến hiện tại và ghi dữ liệu hội tụ.
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best, best_cost = population[index].copy(), float(costs[index])
        history.append(GenerationRecord(generation, best_cost))
    # BƯỚC 5 - KẾT THÚC: thêm kho ở hai đầu để trả hành trình đầy đủ.
    # best_cost đã tính cả đoạn đi từ kho và đoạn quay về kho.
    return OptimizationResult(algorithm, config, [0, *best, 0], best_cost,
                              perf_counter() - started, history, trace)
