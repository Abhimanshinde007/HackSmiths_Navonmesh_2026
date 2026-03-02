import { useEffect, useState, useRef } from "react";
import { apiFetch, apiJson } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Package, Upload, ArrowUpDown, AlertTriangle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface StockItem {
  id: number;
  sku: string;
  name: string;
  current_stock: number;
  safety_stock: number;
}

type SortKey = keyof StockItem;

export default function Inventory() {
  const [stock, setStock] = useState<StockItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortAsc, setSortAsc] = useState(true);
  const [uploading, setUploading] = useState(false);
  const inwardRef = useRef<HTMLInputElement>(null);
  const outwardRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const fetchStock = async () => {
    setLoading(true);
    try {
      const data = await apiJson<StockItem[]>("/inventory/stock");
      setStock(data);
    } catch (err: any) {
      toast({ title: "Error fetching stock", description: err.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchStock(); }, []);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const sorted = [...stock].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    const cmp = typeof av === "string" ? av.localeCompare(bv as string) : (av as number) - (bv as number);
    return sortAsc ? cmp : -cmp;
  });

  const handleUpload = async () => {
    const inwardFiles = inwardRef.current?.files;
    const outwardFiles = outwardRef.current?.files;
    if (!inwardFiles?.length && !outwardFiles?.length) return;

    const form = new FormData();
    if (inwardFiles) Array.from(inwardFiles).forEach(f => form.append("inward_files", f));
    if (outwardFiles) Array.from(outwardFiles).forEach(f => form.append("outward_files", f));

    setUploading(true);
    try {
      await apiFetch("/inventory/upload/stock", { method: "POST", body: form });
      toast({ title: "Stock updated successfully" });
      await fetchStock();
      if (inwardRef.current) inwardRef.current.value = "";
      if (outwardRef.current) outwardRef.current.value = "";
    } catch (err: any) {
      toast({ title: "Upload failed", description: err.message, variant: "destructive" });
    } finally {
      setUploading(false);
    }
  };

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <TableHead
      className="cursor-pointer select-none font-mono text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground"
      onClick={() => handleSort(field)}
    >
      <div className="flex items-center gap-1">
        {label}
        <ArrowUpDown className="h-3 w-3" />
      </div>
    </TableHead>
  );

  const lowStockCount = stock.filter(s => s.current_stock < s.safety_stock).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono text-2xl font-bold tracking-wider text-foreground">INVENTORY STOCK</h1>
          <p className="text-sm text-muted-foreground font-mono">Real-time stock monitoring</p>
        </div>
        {lowStockCount > 0 && (
          <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive font-mono">
            <AlertTriangle className="h-4 w-4" />
            {lowStockCount} items below safety stock
          </div>
        )}
      </div>

      {/* Chart */}
      <Card className="border-border/50 bg-card/80">
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            Stock Levels Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          {stock.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sorted.slice(0, 20)} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 22%)" />
                <XAxis dataKey="sku" tick={{ fill: "hsl(215 15% 55%)", fontSize: 11, fontFamily: "JetBrains Mono" }} />
                <YAxis tick={{ fill: "hsl(215 15% 55%)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "hsl(220 25% 13%)", border: "1px solid hsl(215 20% 22%)", borderRadius: 6, fontFamily: "JetBrains Mono", fontSize: 12 }}
                  labelStyle={{ color: "hsl(210 20% 90%)" }}
                />
                <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 12 }} />
                <Bar dataKey="current_stock" fill="hsl(205 100% 55%)" name="Current Stock" radius={[2, 2, 0, 0]} />
                <Bar dataKey="safety_stock" fill="hsl(0 70% 50%)" name="Safety Stock" radius={[2, 2, 0, 0]} opacity={0.6} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center text-muted-foreground py-8 font-mono text-sm">
              {loading ? "Loading..." : "No stock data"}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Table */}
      <Card className="border-border/50 bg-card/80">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="border-border/50 hover:bg-transparent">
                <SortHeader label="SKU" field="sku" />
                <SortHeader label="Name" field="name" />
                <SortHeader label="Current Stock" field="current_stock" />
                <SortHeader label="Safety Stock" field="safety_stock" />
                <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((item) => {
                const low = item.current_stock < item.safety_stock;
                return (
                  <TableRow key={item.id} className={`border-border/30 ${low ? "bg-destructive/5" : ""}`}>
                    <TableCell className="font-mono text-sm text-primary">{item.sku}</TableCell>
                    <TableCell className="text-sm">{item.name}</TableCell>
                    <TableCell className="font-mono text-sm">{item.current_stock}</TableCell>
                    <TableCell className="font-mono text-sm">{item.safety_stock}</TableCell>
                    <TableCell>
                      {low ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-mono text-destructive">
                          <AlertTriangle className="h-3 w-3" /> LOW
                        </span>
                      ) : (
                        <span className="inline-flex rounded-full bg-primary/10 px-2 py-0.5 text-xs font-mono text-primary">
                          OK
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Upload */}
      <Card className="border-border/50 bg-card/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
            <Upload className="h-4 w-4" />
            Upload Stock Files
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Inward Files</Label>
              <Input ref={inwardRef} type="file" multiple accept=".xlsx,.xls" className="bg-muted/50 font-mono text-sm" />
            </div>
            <div className="space-y-2">
              <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Outward Files</Label>
              <Input ref={outwardRef} type="file" multiple accept=".xlsx,.xls" className="bg-muted/50 font-mono text-sm" />
            </div>
          </div>
          <Button onClick={handleUpload} disabled={uploading} className="font-mono tracking-wider">
            <Package className="h-4 w-4 mr-2" />
            {uploading ? "PROCESSING..." : "UPLOAD & UPDATE"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
