"""Bring current repository pointers in line with the completed seven-policy package."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]

def rewrite(path,fn):
    p=ROOT/path;old=p.read_text('utf-8');new=fn(old)
    if new!=old:p.write_text(new,encoding='utf-8')

def main():
    cs=json.loads((ROOT/'runs/search_seven_confirmation_20261005/summary.json').read_text('utf-8'))
    latest='已实现ILS（迭代局部搜索）、SA（模拟退火）和无交叉AdaLead改编版，与原四法形成七种同起点策略。新冻结比较：30开发种子6001–6030，两代理、1024/4096上限、1260子运行；100确认种子8001–8100，1024上限、1400子运行。七者共同实际预测次数为主表，预设两两共同次数为补充。139项本地测试及逐条档案审计通过；无新真实GBSA、MD、对接或服务器同步。'
    rewrite('PROJECT_STATUS.md',lambda s:s.replace(s.splitlines()[4],
        '本次七算法实施与比较：'+latest+'见[七算法详细结果](docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)和[证据索引](evidence/search_seven_20261005/README.md)。')
        .replace('比较随机游走、爬山、束搜索和无交叉遗传算法的计算行为',
                 '比较随机游走、爬山、束搜索、无交叉遗传算法及新接入的ILS/SA/无交叉AdaLead的计算行为')
        .replace('## 2026-10-05 本轮搜索层收尾','## 2026-10-05 首轮四算法收尾（历史条件保留）'))
    rewrite('README.md',lambda s:s.replace('四种同起点策略（随机游走、爬山、束搜索、仅变异遗传算法）已接通。',
        '七种同起点策略（随机游走、爬山、束搜索、仅变异遗传算法、迭代局部搜索、模拟退火、无交叉AdaLead改编版）已接通。最新七法30开发/100确认共2660子运行，见[七算法结果](docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)。此前四法的历史比较保留：'))
    rewrite('docs/README.md',lambda s:s.replace('ILS、SA和无交叉AdaLead等是尚未实现的条件性建议，不计入已完成实验。',
        'ILS、SA和无交叉AdaLead改编版已实现并完成七法30开发/100确认，详见[七算法实施与结果](汇报/9月/七算法同起点比较与实施结果_20261005.md)与[新冻结方案](项目记录/七算法同起点比较冻结方案_20261005.md)。BO、CbAS和RL仍为后期条件性方法。'))
    rewrite('scripts/README.md',lambda s:s.replace('四种同起点策略：随机游走、爬山、束搜索、`mutation_only_ga`（无交叉遗传算法）；另保留历史分层随机覆盖策略',
        '七种同起点策略：随机游走、爬山、束搜索、无交叉GA、ILS、SA、无交叉AdaLead改编版；另保留历史分层随机覆盖策略')
        .replace('10 月 5 日冻结四算法套件：','10 月 5 日冻结四/七算法套件（显式 --policies）：')
        .replace('当前四算法的正式比较入口为','当前七算法的正式比较入口为')
        .replace('2026-10-05 新增 `mutation_only_ga`：仅变异、无交叉的单父本精英种群策略。',
        '2026-10-05 已支持 `mutation_only_ga`、`iterated_local_search`、`simulated_annealing`、`mutation_only_adalead`；均无交叉。新增三法固定参数、全部表格及可重复实践命令见[七算法实施结果](../docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)。已有七法套件目录需先核验或复制配置另换 out-dir，不能直接写回旧包。'))
    rewrite('AGENTS.md',lambda s:s.replace('- Four matched-start policies (random walk, hill climb, beam, mutation-only GA) share',
        '- Seven matched-start policies (random walk, hill climb, beam, mutation-only GA, iterated local search, simulated annealing, mutation-only AdaLead adaptation) share')
        .replace('- Frozen local suite: development401–430,',
        '- Latest seven-policy suite: development6001–6030, two proxies, budgets1024/4096,1260subruns including420refits; confirmation8001–8100,budget1024,1400subruns. Generation guard256 for all; primary seven-way common actual-call prefixes and pre-specified pairwise companion are distinct. No crossover in any new policy. AdaLead uses additive signed-minimization tolerance and root-relative rollouts; SA global dedup is an optimization variant, not equilibrium sampling. See `docs/汇报/9月/七算法同起点比较与实施结果_20261005.md` and `evidence/search_seven_20261005/` for actual values.139local tests pass.\n- Earlier four-policy suite (retained): development401–430,'))
    rewrite('CHANGELOG.md',lambda s:s.replace('# Change log\n',
        '# Change log\n\n## 2026-10-05（三个新策略实现及七算法统一比较）\n\n- '+latest+'\n- 新增独立操作配置、七算法冻结方案与完整生工结果汇报；总汇报§4.15、阶段图及索引更新。原四算法的旧证据不改写。确认启动前因开发重训种子重叠，审计调整确认编号7001–7100为8001–8100；算法/参数/预算/比较对未改。\n- 新方法精确恢复、负分阈值、退火概率、扰动机制与未来暂停分组交接均测试；独立审计重算主/两两共同次数最低预测。结果仍限固定代理景观，真实GBSA算法优胜未判定。\n'))
    rewrite('evidence/README.md',lambda s:s+'\n\n## 2026-10-05 七算法扩展\n\n[七算法证据包](search_seven_20261005/README.md)：ILS、SA、无交叉AdaLead改编版与原四策略，30开发/100确认，2660正式子运行；主/补充预算口径分列，139本地测试与逐次审计。无新真实标签。旧[四算法包](search_suite_20261005/README.md)保留。\n')
    guide='docs/项目记录/无交叉遗传算法运行与阶段B收尾操作指南_20261005.md'
    rewrite(guide,lambda s:s.replace('搜索层在固定 GBSA 评分器上运行，已支持同起点随机游走、多起点爬山、束搜索和仅变异遗传算法。',
        '搜索层在固定 GBSA 评分器上运行，现支持七种：同起点随机游走、多起点爬山、束搜索、仅变异遗传算法、迭代局部搜索、模拟退火、无交叉AdaLead改编版。')
        .replace('ILS、SA、无交叉AdaLead等尚未实现，不能将现有命令的策略名称直接改成这些名字就视为已经支持。',
        'ILS、SA、无交叉AdaLead改编版现已支持，并完成七法统一30开发/100确认。新参数、全部表格与直接运行配置见[七算法完整实施结果](../汇报/9月/七算法同起点比较与实施结果_20261005.md)§2、§8。原四法命令及结果在本文保留历史条件。')
        +'\n\n## 8. 本轮新增三法：直接实践入口\n\n最新七法冻结方案见[七算法计划](七算法同起点比较冻结方案_20261005.md)，完整读数与选择规则见[七算法结果](../汇报/9月/七算法同起点比较与实施结果_20261005.md)。\n\n```powershell\npython scripts/run_experiment.py configs/experiments/search_ils_demo_20261005.json\npython scripts/run_experiment.py configs/experiments/search_sa_demo_20261005.json\npython scripts/run_experiment.py configs/experiments/search_adalead_demo_20261005.json\npython tools/verify_search_suite.py runs/search_seven_development_20261005\npython tools/verify_search_suite.py runs/search_seven_confirmation_20261005\n```\n\n三个单次配置由运行器自动新建包，可重复实践；已完成套件有固定输出目录，重算须先复制配置并换out-dir，具体示例见七算法结果§8.3。无真实测量请求。\n')
    print('current indexes updated')

if __name__=='__main__':main()
