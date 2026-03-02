import { useState, useRef } from "react";
import { apiFetch, apiJson } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Upload, TrendingUp, Users, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Forecast() {
  const [salesData, setSalesData] = useState<any[] | null>(null);
  const [forecastResult, setForecastResult] = useState<any | null>(null);
  const [predictionDays, setPredictionDays] = useState(90);
  const [uploadingState, setUploadingState] = useState<"idle" | "uploading" | "forecasting">("idle");
  const filesRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleUploadSales = async () => {
    const files = filesRef.current?.files;
    if (!files?.length) return;

    const form = new FormData();
    Array.from(files).forEach(f => form.append("files", f));

    setUploadingState("uploading");
    try {
      const res = await apiFetch("/inventory/upload/sales", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setSalesData(json.data);
      setForecastResult(null);
      toast({ title: "Sales data parsed", description: `${json.data.length} records loaded` });
    } catch (err: any) {
      toast({ title: "Upload failed", description: err.message, variant: "destructive" });
    } finally {
      setUploadingState("idle");
    }
  };

  const handleRunForecast = async () => {
    if (!salesData) return;
    setUploadingState("forecasting");
    try {
      const result = await apiJson<any>("/forecast/run", {
        method: "POST",
        body: JSON.stringify({ sales_data: salesData, prediction_window_days: predictionDays }),
      });
      setForecastResult(result);
      toast({ title: "Forecast complete" });
    } catch (err: any) {
      toast({ title: "Forecast failed", description: err.message, variant: "destructive" });
    } finally {
      setUploadingState("idle");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-mono text-2xl font-bold tracking-wider text-foreground">DEMAND FORECAST</h1>
        <p className="text-sm text-muted-foreground font-mono">Predictive demand analysis engine</p>
      </div>

      {/* Step 1: Upload */}
      <Card className="border-border/50 bg-card/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
            <Upload className="h-4 w-4" />
            Step 1 — Upload Sales Data
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Sales Excel Files</Label>
            <Input ref={filesRef} type="file" multiple accept=".xlsx,.xls" className="bg-muted/50 font-mono text-sm" />
          </div>
          <Button onClick={handleUploadSales} disabled={uploadingState === "uploading"} className="font-mono tracking-wider">
            {uploadingState === "uploading" ? "PARSING..." : "UPLOAD SALES DATA"}
          </Button>
          {salesData && (
            <p className="text-sm text-primary font-mono">✓ {salesData.length} sales records loaded</p>
          )}
        </CardContent>
      </Card>

      {/* Step 2: Run Forecast */}
      {salesData && (
        <Card className="border-border/50 bg-card/80">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
              <TrendingUp className="h-4 w-4" />
              Step 2 — Run Forecast
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-end gap-4">
              <div className="space-y-2">
                <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                  Prediction Window (days)
                </Label>
                <Input
                  type="number"
                  value={predictionDays}
                  onChange={(e) => setPredictionDays(Number(e.target.value))}
                  className="w-32 bg-muted/50 font-mono"
                />
              </div>
              <Button onClick={handleRunForecast} disabled={uploadingState === "forecasting"} className="font-mono tracking-wider">
                {uploadingState === "forecasting" ? "COMPUTING..." : "RUN FORECAST"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {forecastResult && (
        <>
          {/* Errors */}
          {forecastResult.errors?.length > 0 && (
            <Card className="border-destructive/50 bg-destructive/5">
              <CardContent className="pt-6">
                <div className="space-y-2">
                  {forecastResult.errors.map((err: string, i: number) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-destructive font-mono">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      {err}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Anchor Customers */}
          {forecastResult.anchor_customers?.length > 0 && (
            <Card className="border-border/50 bg-card/80">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
                  <Users className="h-4 w-4" />
                  Anchor Customers
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/50 hover:bg-transparent">
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Customer</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Total Qty</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Invoices</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Share %</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {forecastResult.anchor_customers.map((c: any, i: number) => (
                      <TableRow key={i} className="border-border/30">
                        <TableCell className="font-mono text-sm text-primary">{c.Customer}</TableCell>
                        <TableCell className="font-mono text-sm">{c['Total Qty']}</TableCell>
                        <TableCell className="font-mono text-sm">{c.Invoices}</TableCell>
                        <TableCell className="font-mono text-sm">{c['Share %']}%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {/* Demand Forecasts */}
          {forecastResult.forecast?.length > 0 && (
            <Card className="border-border/50 bg-card/80">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-muted-foreground">
                  <TrendingUp className="h-4 w-4" />
                  Demand Predictions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/50 hover:bg-transparent">
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Customer</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Predicted Product</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Predicted Date</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Frequency</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {forecastResult.forecast.map((f: any, i: number) => (
                      <TableRow key={i} className="border-border/30">
                        <TableCell className="font-mono text-sm text-primary">{f.Customer}</TableCell>
                        <TableCell className="font-mono text-sm">{f['Predicted Product']}</TableCell>
                        <TableCell className="font-mono text-sm">{f['Predicted Next Order']}</TableCell>
                        <TableCell className="font-mono text-sm">~{f['Avg Interval (Days)']} days</TableCell>
                        <TableCell className="font-mono text-sm">{f['Confidence %']}%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {/* Material Requirements */}
          {forecastResult.material_requirements?.length > 0 && (
            <Card className="border-border/50 bg-card/80">
              <CardHeader>
                <CardTitle className="font-mono text-sm uppercase tracking-wider text-primary">
                  Material Requirements & Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-border/50 hover:bg-transparent">
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Material</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Required For</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Need</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">In Stock</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Shortfall</TableHead>
                      <TableHead className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {forecastResult.material_requirements.map((mr: any, i: number) => {
                      const isDanger = mr.Status === 'BUY NOW' || mr.Status === 'CHECK STOCK';
                      return (
                        <TableRow key={i} className={`border-border/30 ${isDanger ? "bg-destructive/5" : ""}`}>
                          <TableCell className="font-mono text-sm font-bold text-foreground">{mr.Material}</TableCell>
                          <TableCell className="font-mono text-sm">{mr.Customer}</TableCell>
                          <TableCell className="font-mono text-sm">{mr.Need}</TableCell>
                          <TableCell className="font-mono text-sm">{mr['In Stock']}</TableCell>
                          <TableCell className="font-mono text-sm text-destructive">{mr.Shortfall || '-'}</TableCell>
                          <TableCell>
                            <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-mono ${isDanger ? "bg-destructive/10 text-destructive" : "bg-primary/10 text-primary"}`}>
                              {mr.Status}
                            </span>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
