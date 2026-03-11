const ACTION_COLORS = {
  Walking: "text-blue-400",
  Fall: "text-red-400",
  Standing: "text-green-400",
  Sitting: "text-amber-400",
};

/**
 * Live table of tracked persons.
 * Props: tracks — [{ id, action, conf, bbox }]
 */
export default function TrackTable({ tracks }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-slate-300">Tracked Persons</h3>
      {tracks.length === 0 ? (
        <p className="text-slate-500 text-xs">No tracks</p>
      ) : (
        <div className="overflow-auto max-h-48 rounded-lg border border-slate-700">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-800 text-slate-400 sticky top-0">
              <tr>
                <th className="px-3 py-2">ID</th>
                <th className="px-3 py-2">Action</th>
                <th className="px-3 py-2">Conf</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {tracks.map((t) => (
                <tr
                  key={t.id}
                  className="hover:bg-slate-800/50 transition-colors"
                >
                  <td className="px-3 py-1.5 font-mono text-slate-300">
                    #{t.id}
                  </td>
                  <td
                    className={`px-3 py-1.5 font-medium ${ACTION_COLORS[t.action] ?? "text-slate-200"}`}
                  >
                    {t.action}
                  </td>
                  <td className="px-3 py-1.5 text-slate-400 font-mono">
                    {(t.conf * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
